package vitencoder

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
)

// Depth Anything V2 (Yang et al. 2024; its Apache-2.0 code, as comfyui_controlnet_aux carries it) reads a picture's
// relative depth: a DINOv2 encoder's tokens after four of its blocks, each through the final norm, turned by a DPT head
// into a map, larger where the scene is nearer. The encoder is this package's; the head is below.

// depthSpec is a depth checkpoint's head, from its reviewed declaration: the blocks whose tokens it reads, its fusion
// width, each level's projected width, the side its input is sized to at least, the offset DINOv2 adds to the grid when
// it resamples its positions, and the mean and spread pixels are normalized by.
type depthSpec struct {
	Layers         []int      `json:"layers"`
	Features       int        `json:"features"`
	OutChannels    [4]int     `json:"out_channels"`
	InputSize      int        `json:"input_size"`
	PositionOffset float64    `json:"position_offset"`
	Mean           [3]float64 `json:"mean"`
	Std            [3]float64 `json:"std"`
}

// depthConv is a convolution of the head: its weight [out][in][kernel][kernel] (a transposed one's [in][out][k][k]),
// its bias or none, and its stride and padding.
type depthConv struct {
	weight, bias            []float32
	in, out, kernel, stride int
	padding                 int
	transposed              bool
}

// residualUnit is a fusion block's residual convolution unit: x + conv2(relu(conv1(relu(x)))).
type residualUnit struct{ conv1, conv2 depthConv }

// fusionBlock is a DPT fusion block: an input, plus the residual unit of a second where it has one, through a second
// residual unit, resized (bilinear, corners aligned) and projected.
type fusionBlock struct {
	unit1, unit2 residualUnit
	out          depthConv
}

// DepthModel is a loaded Depth Anything V2 checkpoint.
type DepthModel struct {
	encoder *Encoder
	spec    depthSpec
	// trained is the side of the position grid the checkpoint holds.
	trained int
	// projects, resize and levels are each level's projection, resizing (transposed 4 and 2, none, a stride-2
	// convolution) and convolution to the fusion width; refine fuses the levels, deepest first at index 3.
	projects, levels [4]depthConv
	resize           [4]*depthConv
	refine           [4]fusionBlock
	output1, output2 depthConv
	output3          depthConv
}

// LoadDepthAnything reads a Depth Anything V2 PyTorch checkpoint (depth_anything_v2_vits.pth) by its reviewed
// declaration, which states the encoder's geometry the file does not.
func LoadDepthAnything(path string) (*DepthModel, error) {
	declaration, known, err := declarationFor(depthAnythingV2)
	if err != nil || !known || declaration.Geometry == nil || declaration.Depth == nil {
		return nil, errors.Join(err, errors.New("vitencoder: Depth Anything V2's declaration states no geometry and head"))
	}
	g, spec := *declaration.Geometry, *declaration.Depth
	if g.PatchSize <= 0 || g.Hidden <= 0 || g.Heads <= 0 || g.Hidden%g.Heads != 0 || g.Layers <= 0 || g.Epsilon <= 0 || g.Registers != 0 ||
		g.ImageSize%g.PatchSize != 0 || len(spec.Layers) != len(spec.OutChannels) || spec.Features <= 0 || spec.InputSize%g.PatchSize != 0 {
		return nil, errors.New("vitencoder: Depth Anything V2's declaration is incomplete")
	}
	for _, index := range spec.Layers {
		if index < 0 || index >= g.Layers {
			return nil, fmt.Errorf("vitencoder: the head reads block %d of %d", index, g.Layers)
		}
	}
	catalog, err := pytorchzip.ReadCatalog(path)
	if err != nil {
		return nil, err
	}
	reader, err := pytorchzip.Open(path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	read := func(name string, size int) ([]float32, error) {
		bindings, err := pytorchzip.CompileBindings(catalog.Tensors, []string{name})
		if err != nil {
			return nil, fmt.Errorf("vitencoder: tensor %s: %w", name, err)
		}
		values, err := reader.ReadBinding(bindings[0])
		if err != nil {
			return nil, fmt.Errorf("vitencoder: tensor %s: %w", name, err)
		}
		if size != checked.UnknownCount() && len(values) != size {
			return nil, fmt.Errorf("vitencoder: tensor %s holds %d values, want %d", name, len(values), size)
		}
		return values, nil
	}
	trained := g.ImageSize / g.PatchSize
	e := &Encoder{geometry: g, patchInputSize: channels * g.PatchSize * g.PatchSize}
	t := declaration.Tensors
	for _, binding := range []struct {
		target *[]float32
		name   string
		size   int
	}{
		{&e.classToken, t.ClassToken, g.Hidden},
		{&e.position, t.PositionEmbeddings, (1 + trained*trained) * g.Hidden},
		{&e.patchWeight, t.PatchWeight, g.Hidden * e.patchInputSize},
		{&e.patchBias, t.PatchBias, g.Hidden},
		{&e.finalWeight, t.FinalNormWeight, g.Hidden},
		{&e.finalBias, t.FinalNormBias, g.Hidden},
	} {
		if *binding.target, err = read(binding.name, binding.size); err != nil {
			return nil, err
		}
	}
	if e.layers, err = bindLayers(read, g, t.Layer); err != nil {
		return nil, err
	}
	m := &DepthModel{encoder: e, spec: spec, trained: trained}
	conv := func(name string, in, out, kernel, stride, padding int, bias, transposed bool) (depthConv, error) {
		c := depthConv{in: in, out: out, kernel: kernel, stride: stride, padding: padding, transposed: transposed}
		var err error
		if c.weight, err = read(name+".weight", in*out*kernel*kernel); err != nil {
			return c, err
		}
		if bias {
			c.bias, err = read(name+".bias", out)
		}
		return c, err
	}
	unit := func(name string) (residualUnit, error) {
		conv1, err := conv(name+".conv1", spec.Features, spec.Features, 3, 1, 1, true, false)
		if err != nil {
			return residualUnit{}, err
		}
		conv2, err := conv(name+".conv2", spec.Features, spec.Features, 3, 1, 1, true, false)
		return residualUnit{conv1, conv2}, err
	}
	for level, width := range spec.OutChannels {
		if m.projects[level], err = conv(fmt.Sprintf("depth_head.projects.%d", level), g.Hidden, width, 1, 1, 0, true, false); err != nil {
			return nil, err
		}
		name := fmt.Sprintf("depth_head.resize_layers.%d", level)
		var resize depthConv
		switch level {
		case 0:
			resize, err = conv(name, width, width, 4, 4, 0, true, true)
		case 1:
			resize, err = conv(name, width, width, 2, 2, 0, true, true)
		case 3:
			resize, err = conv(name, width, width, 3, 2, 1, true, false)
		}
		if err != nil {
			return nil, err
		}
		if level != 2 {
			m.resize[level] = &resize
		}
		if m.levels[level], err = conv(fmt.Sprintf("depth_head.scratch.layer%d_rn", level+1), width, spec.Features, 3, 1, 1, false, false); err != nil {
			return nil, err
		}
		name = fmt.Sprintf("depth_head.scratch.refinenet%d", level+1)
		var block fusionBlock
		if block.unit1, err = unit(name + ".resConfUnit1"); err != nil {
			return nil, err
		}
		if block.unit2, err = unit(name + ".resConfUnit2"); err != nil {
			return nil, err
		}
		if block.out, err = conv(name+".out_conv", spec.Features, spec.Features, 1, 1, 0, true, false); err != nil {
			return nil, err
		}
		m.refine[level] = block
	}
	if m.output1, err = conv("depth_head.scratch.output_conv1", spec.Features, spec.Features/2, 3, 1, 1, true, false); err != nil {
		return nil, err
	}
	if m.output2, err = conv("depth_head.scratch.output_conv2.0", spec.Features/2, 32, 3, 1, 1, true, false); err != nil {
		return nil, err
	}
	if m.output3, err = conv("depth_head.scratch.output_conv2.2", 32, 1, 1, 1, 0, true, false); err != nil {
		return nil, err
	}
	return m, nil
}

// Depth is the relative depth of normalized channel-first pixels of height by width, each a multiple of the patch
// side: a map of height by width, larger where the scene is nearer, as the model's forward gives it (max_depth 1).
func (m *DepthModel) Depth(pixels []float32, height, width int) ([]float32, error) {
	g := m.encoder.geometry
	if height <= 0 || width <= 0 || height%g.PatchSize != 0 || width%g.PatchSize != 0 || len(pixels) != channels*height*width {
		return nil, fmt.Errorf("vitencoder: %d pixels are not a %d x %d picture of whole patches", len(pixels), width, height)
	}
	rows, columns := height/g.PatchSize, width/g.PatchSize
	position := dinoPositions(m.encoder.position, m.trained, rows, columns, g.Hidden, m.spec.PositionOffset)
	tokens := 1 + rows*columns
	hidden := m.encoder.embed(pixels, rows, columns, position)
	taken := make([][]float32, 0, len(m.spec.Layers))
	m.encoder.blocks(hidden, tokens, func(index int) {
		for _, wanted := range m.spec.Layers {
			if wanted == index {
				normed := make([]float32, len(hidden))
				hostmath.LayerNormInto(normed, hidden, m.encoder.finalWeight, m.encoder.finalBias, tokens, g.Hidden, g.Epsilon)
				taken = append(taken, normed[g.Hidden:])
			}
		}
	})
	if len(taken) != len(m.spec.Layers) {
		return nil, errors.New("vitencoder: the encoder ran fewer blocks than the head reads")
	}
	var level [4]plane
	for index, patches := range taken {
		// Each block's patch tokens as a channel-first plane over the grid.
		x := plane{channels: g.Hidden, height: rows, width: columns, values: make([]float32, len(patches))}
		for token := range rows * columns {
			for channel := range g.Hidden {
				x.values[channel*rows*columns+token] = patches[token*g.Hidden+channel]
			}
		}
		var err error
		if x, err = m.projects[index].apply(x); err != nil {
			return nil, err
		}
		if m.resize[index] != nil {
			if x, err = m.resize[index].apply(x); err != nil {
				return nil, err
			}
		}
		if level[index], err = m.levels[index].apply(x); err != nil {
			return nil, err
		}
	}
	path, err := m.refine[3].fuse(level[3], nil, level[2].height, level[2].width)
	for index := 2; index >= 0 && err == nil; index-- {
		next := level[index]
		height, width := 2*next.height, 2*next.width
		if index > 0 {
			height, width = level[index-1].height, level[index-1].width
		}
		path, err = m.refine[index].fuse(path, &next, height, width)
	}
	if err != nil {
		return nil, err
	}
	out, err := m.output1.apply(path)
	if err != nil {
		return nil, err
	}
	out = out.resized(height, width)
	if out, err = m.output2.apply(out); err != nil {
		return nil, err
	}
	out.relu()
	if out, err = m.output3.apply(out); err != nil {
		return nil, err
	}
	out.relu()
	return finite(out.values)
}

// DepthMap is the depth map ComfyUI's DepthAnythingV2Preprocessor (comfyui_controlnet_aux, Depth Anything V2 -
// Relative) makes of a picture at a detect resolution, white for near: of the picture, height by width RGB values in
// the unit range as ComfyUI holds an image, a row's pixels after another's. As the node runs it: the picture read as
// 8-bit levels, 255 x truncated; its levels over 255 in float64 resized bicubically (OpenCV) to a multiple of the patch
// side whose shorter side is at least the model's input size, normalized and read by the model; the depth resized back
// bilinearly with corners aligned, stretched to 0..255 in float32 and truncated; then resized so its shorter side is
// the resolution, bicubically (OpenCV, 8-bit) where that enlarges it and by area where it shrinks it. It returns the
// map's levels and its height and width.
func (m *DepthModel) DepthMap(unit []float32, height, width, resolution int) ([]uint8, int, int, error) {
	return m.depthMap(unit, height, width, resolution, m.Depth)
}

// depthMap is DepthMap with the model's depth of its normalized input given by depth, the host's (Depth) or the
// card's (DepthCard).
func (m *DepthModel) depthMap(unit []float32, height, width, resolution int, depth func(pixels []float32, height, width int) ([]float32, error)) ([]uint8, int, int, error) {
	if height <= 0 || width <= 0 || resolution <= 0 || len(unit) != height*width*channels {
		return nil, 0, 0, fmt.Errorf("vitencoder: %d values are not a %d x %d RGB picture", len(unit), width, height)
	}
	values := make([]float64, len(unit))
	for index, value := range unit {
		values[index] = float64(uint8(min(max(value*255, 0), 255))) / 255
	}
	inputHeight, inputWidth := m.inputSize(height, width)
	resized, err := media.CubicResizeFloat64(values, height, width, channels, inputHeight, inputWidth)
	if err != nil {
		return nil, 0, 0, err
	}
	area := inputHeight * inputWidth
	pixels := make([]float32, len(resized))
	for position := range area {
		for c := range channels {
			pixels[c*area+position] = float32((resized[position*channels+c] - m.spec.Mean[c]) / m.spec.Std[c])
		}
	}
	estimate, err := depth(pixels, inputHeight, inputWidth)
	if err != nil {
		return nil, 0, 0, err
	}
	full := onePlane(estimate, inputHeight, inputWidth).resized(height, width).values
	lowest, highest := full[0], full[0]
	for _, value := range full {
		lowest, highest = min(lowest, value), max(highest, value)
	}
	levels := make([]uint8, len(full))
	for index, value := range full {
		levels[index] = uint8((value - lowest) / (highest - lowest) * 255)
	}
	scale := float64(resolution) / float64(min(height, width))
	mapHeight, mapWidth := int(math.RoundToEven(float64(height)*scale)), int(math.RoundToEven(float64(width)*scale))
	var detected []uint8
	if scale > 1 {
		detected, err = media.CubicResizeLevels(levels, height, width, 1, mapHeight, mapWidth)
	} else {
		detected, err = media.AreaResizeLevels(levels, height, width, 1, mapHeight, mapWidth)
	}
	return detected, mapHeight, mapWidth, err
}

// inputSize is the model's input for a picture of height by width, as Depth Anything V2's Resize sizes it
// (keep_aspect_ratio, lower_bound, ensure_multiple_of the patch side): both sides scaled by the larger of the input
// size over each, each rounded half to even to a multiple of the patch side, and rounded up where that falls below the
// input size.
func (m *DepthModel) inputSize(height, width int) (int, int) {
	side, multiple := float64(m.spec.InputSize), float64(m.encoder.geometry.PatchSize)
	scale := max(side/float64(height), side/float64(width))
	constrain := func(x float64) int {
		y := math.RoundToEven(x/multiple) * multiple
		if y < side {
			y = math.Ceil(x/multiple) * multiple
		}
		return int(y)
	}
	return constrain(scale * float64(height)), constrain(scale * float64(width))
}

// onePlane is one channel of height by width values as a plane.
func onePlane(values []float32, height, width int) plane {
	return plane{channels: 1, height: height, width: width, values: values}
}

// plane is a channel-first activation: channels planes of height rows of width values.
type plane struct {
	channels, height, width int
	values                  []float32
}

// relu clamps the plane's negative values to 0, in place.
func (p plane) relu() {
	for index, value := range p.values {
		p.values[index] = max(value, 0)
	}
}

// apply is the convolution of x: a plain one through hostmath's convolution (one frame, f64 sums), a transposed one,
// whose kernel is its stride, each input value spread over its own block of outputs.
func (c depthConv) apply(x plane) (plane, error) {
	if x.channels != c.in {
		return plane{}, fmt.Errorf("vitencoder: a convolution of %d inputs given %d channels", c.in, x.channels)
	}
	if c.transposed {
		if c.kernel != c.stride || c.padding != 0 {
			return plane{}, errors.New("vitencoder: a transposed convolution whose kernel is not its stride")
		}
		out := plane{channels: c.out, height: x.height * c.kernel, width: x.width * c.kernel}
		out.values = make([]float32, out.channels*out.height*out.width)
		area := x.height * x.width
		hostmath.ParallelRangeF64(c.out, area*c.in*c.kernel*c.kernel, func(start, end int) {
			for o := start; o < end; o++ {
				for y := range x.height {
					for xx := range x.width {
						for ky := range c.kernel {
							for kx := range c.kernel {
								sum := 0.0
								if c.bias != nil {
									sum = float64(c.bias[o])
								}
								for i := range c.in {
									sum += float64(x.values[i*area+y*x.width+xx]) * float64(c.weight[((i*c.out+o)*c.kernel+ky)*c.kernel+kx])
								}
								out.values[(o*out.height+y*c.kernel+ky)*out.width+xx*c.kernel+kx] = float32(sum)
							}
						}
					}
				}
			}
		})
		return out, nil
	}
	shape := hostmath.Conv3DShape{CIn: c.in, COut: c.out, InT: 1, InH: x.height, InW: x.width, KT: 1, KH: c.kernel, KW: c.kernel,
		PadH: c.padding, PadW: c.padding, StrideT: 1, StrideH: c.stride, StrideW: c.stride}
	_, height, width, err := shape.OutputDims()
	if err != nil {
		return plane{}, err
	}
	out := plane{channels: c.out, height: height, width: width, values: make([]float32, c.out*height*width)}
	return out, hostmath.CausalConv3DInto(out.values, x.values, nil, c.weight, c.bias, 0, shape)
}

// apply is the residual unit of x: x + conv2(relu(conv1(relu(x)))).
func (u residualUnit) apply(x plane) (plane, error) {
	inner := plane{channels: x.channels, height: x.height, width: x.width, values: append([]float32(nil), x.values...)}
	inner.relu()
	inner, err := u.conv1.apply(inner)
	if err != nil {
		return plane{}, err
	}
	inner.relu()
	if inner, err = u.conv2.apply(inner); err != nil {
		return plane{}, err
	}
	for index, value := range x.values {
		inner.values[index] += value
	}
	return inner, nil
}

// fuse is the fusion block over path, plus the residual unit of level where it has one, through its second residual
// unit, resized to height by width (bilinear, corners aligned) and projected.
func (b fusionBlock) fuse(path plane, level *plane, height, width int) (plane, error) {
	if level != nil {
		residual, err := b.unit1.apply(*level)
		if err != nil {
			return plane{}, err
		}
		if len(residual.values) != len(path.values) {
			return plane{}, errors.New("vitencoder: a fusion's two inputs differ in size")
		}
		for index, value := range residual.values {
			path.values[index] += value
		}
	}
	fused, err := b.unit2.apply(path)
	if err != nil {
		return plane{}, err
	}
	return b.out.apply(fused.resized(height, width))
}

// resized is the plane resized to height by width as torch's bilinear interpolation with corners aligned resizes it
// (upsample_bilinear2d): each output at the input position out * (in - 1) / (out - 1), its scale in float32, the four
// neighbours weighted by the fractions, in float32.
func (p plane) resized(height, width int) plane {
	out := plane{channels: p.channels, height: height, width: width, values: make([]float32, p.channels*height*width)}
	scale := func(in, out int) float32 {
		if out > 1 {
			return float32(in-1) / float32(out-1)
		}
		return 0
	}
	rowScale, columnScale := scale(p.height, height), scale(p.width, width)
	hostmath.ParallelRangeF64(p.channels, height*width*4, func(start, end int) {
		for c := start; c < end; c++ {
			source := p.values[c*p.height*p.width:]
			for y := range height {
				real := rowScale * float32(y)
				top := int(real)
				below := 0
				if top < p.height-1 {
					below = 1
				}
				lowWeight := real - float32(top)
				highWeight := 1 - lowWeight
				for x := range width {
					realX := columnScale * float32(x)
					left := int(realX)
					right := 0
					if left < p.width-1 {
						right = 1
					}
					rightWeight := realX - float32(left)
					leftWeight := 1 - rightWeight
					upper := source[top*p.width+left:]
					lower := source[(top+below)*p.width+left:]
					out.values[(c*height+y)*width+x] = highWeight*(leftWeight*upper[0]+rightWeight*upper[right]) +
						lowWeight*(leftWeight*lower[0]+rightWeight*lower[right])
				}
			}
		}
	})
	return out
}

// dinoPositions is the checkpoint's positions for a grid of rows by columns patches, as DINOv2's
// interpolate_pos_encoding resamples them: the class position kept, and the trained grid's patch positions resampled
// bicubically by the scale factors (rows + offset) / trained and (columns + offset) / trained, as torch's
// upsample_bicubic2d resamples without antialiasing (A = -0.75, the source of each output read through the inverse of
// its factor, edges clamped, the rows' four values interpolated across, then down, in float32). A grid of the trained
// side is the checkpoint's own.
func dinoPositions(position []float32, trained, rows, columns, hidden int, offset float64) []float32 {
	if rows == trained && columns == trained {
		return position
	}
	resampled := make([]float32, (1+rows*columns)*hidden)
	copy(resampled, position[:hidden])
	rowInverse := float32(1 / ((float64(rows) + offset) / float64(trained)))
	columnInverse := float32(1 / ((float64(columns) + offset) / float64(trained)))
	type taps struct {
		first   int
		weights [4]float32
	}
	tapsOf := func(inverse float32, index int) taps {
		real := inverse*(float32(index)+0.5) - 0.5
		first := int(math.Floor(float64(real)))
		t := real - float32(first)
		return taps{first: first, weights: cubicWeights(t)}
	}
	clamp := func(index int) int { return min(max(index, 0), trained-1) }
	patches := position[hidden:]
	hostmath.ParallelRangeF64(rows, columns*hidden*16, func(start, end int) {
		for row := start; row < end; row++ {
			down := tapsOf(rowInverse, row)
			for column := range columns {
				across := tapsOf(columnInverse, column)
				into := resampled[(1+row*columns+column)*hidden:][:hidden]
				for channel := range hidden {
					var lines [4]float32
					for k := range 4 {
						line := clamp(down.first - 1 + k)
						for j := range 4 {
							lines[k] += patches[(line*trained+clamp(across.first-1+j))*hidden+channel] * across.weights[j]
						}
					}
					into[channel] = lines[0]*down.weights[0] + lines[1]*down.weights[1] + lines[2]*down.weights[2] + lines[3]*down.weights[3]
				}
			}
		}
	})
	return resampled
}

// cubicWeights are torch's cubic convolution weights (A = -0.75) for the four taps around a fraction t, in float32.
func cubicWeights(t float32) [4]float32 {
	const a = -0.75
	near := func(x float32) float32 { return ((a+2)*x-(a+3))*x*x + 1 }
	far := func(x float32) float32 { return ((a*x-5*a)*x+8*a)*x - 4*a }
	return [4]float32{far(t + 1), near(t), near(1 - t), far(2 - t)}
}
