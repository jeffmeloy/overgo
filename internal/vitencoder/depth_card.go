package vitencoder

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// DepthCard is a Depth Anything V2 model whose encoder blocks and DPT head run on a CUDA device, each input size's
// program compiled the first time the size is asked for and its weights put there with it in F32 (moss-depth-card):
// the host's estimate took 7.8 s of a depth control's map at 1024 x 576, its attention and the head's convolutions
// about 90% of 24 cores. The host prepares and embeds the picture and resizes the estimate as the host's DepthMap does;
// the card gives the depth Depth gives, at F32's rounding.
type DepthCard struct {
	cardDevice
	model    *DepthModel
	mu       sync.Mutex
	programs map[[2]int]*depthProgram
}

// depthProgram is one input size's program: the embedded rows it reads and the depth it gives.
type depthProgram struct {
	compiled        *executor.CompiledGraph
	inputs          *executor.DeviceInputs
	embedded, depth *tensor.Tensor
}

// OpenCard puts the model on the device of ordinal.
func (m *DepthModel) OpenCard(ordinal int) (*DepthCard, error) {
	opened, err := openCardDevice(ordinal)
	if err != nil {
		return nil, err
	}
	return &DepthCard{cardDevice: opened, model: m, programs: map[[2]int]*depthProgram{}}, nil
}

// DepthMap is the model's DepthMap with its depth read on the card.
func (c *DepthCard) DepthMap(ctx context.Context, unit []float32, height, width, resolution int) ([]uint8, int, int, error) {
	return c.model.depthMap(unit, height, width, resolution, func(pixels []float32, height, width int) ([]float32, error) {
		return c.depth(ctx, pixels, height, width)
	})
}

// depth is the relative depth of normalized channel-first pixels as Depth gives it, read on the card.
func (c *DepthCard) depth(ctx context.Context, pixels []float32, height, width int) ([]float32, error) {
	m := c.model
	g := m.encoder.geometry
	if height <= 0 || width <= 0 || height%g.PatchSize != 0 || width%g.PatchSize != 0 || len(pixels) != channels*height*width {
		return nil, fmt.Errorf("vitencoder: %d pixels are not a %d x %d picture of whole patches", len(pixels), width, height)
	}
	program, err := c.program(ctx, height, width)
	if err != nil {
		return nil, err
	}
	rows, columns := height/g.PatchSize, width/g.PatchSize
	position := dinoPositions(m.encoder.position, m.trained, rows, columns, g.Hidden, m.spec.PositionOffset)
	embedded := m.encoder.embed(pixels, rows, columns, position)
	out, err := c.executor.ExecuteCompiled(ctx, program.compiled,
		map[*tensor.Tensor]reference.Value{program.embedded: {Shape: program.embedded.Shape, Data: embedded}}, program.inputs)
	if err != nil {
		return nil, fmt.Errorf("vitencoder depth card: %w", err)
	}
	return finite(out[program.depth].Data)
}

// program is the compiled program of an input of height by width, compiled the first time that size is asked for.
func (c *DepthCard) program(ctx context.Context, height, width int) (*depthProgram, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if program, held := c.programs[[2]int{height, width}]; held {
		return program, nil
	}
	program, weights, err := c.model.cardGraph(height, width)
	if err != nil {
		return nil, err
	}
	if program.compiled, err = executor.Compile(program.depth); err != nil {
		return nil, fmt.Errorf("vitencoder depth card: %w", err)
	}
	if program.inputs, err = c.upload(ctx, program.compiled, weights); err != nil {
		return nil, err
	}
	c.programs[[2]int{height, width}] = program
	return program, nil
}

// cardGraph is the graph of the depth of an input of height by width, the embedded rows it reads and the depth it
// gives, and the weights it reads.
func (m *DepthModel) cardGraph(height, width int) (*depthProgram, cardWeights, error) {
	g := m.encoder.geometry
	rows, columns := uint64(height/g.PatchSize), uint64(width/g.PatchSize)
	hidden, tokens := uint64(g.Hidden), 1+rows*columns
	b, weights := tensor.NewBuilder(), cardWeights{}
	failed := func() error {
		if err := b.Err(); err != nil {
			return fmt.Errorf("vitencoder depth card: %w", err)
		}
		return nil
	}
	program := &depthProgram{embedded: b.Input("embedded", dtype.F32, tensor.MustShape(hidden, tokens))}
	finalWeight, finalBias := weights.input(b, "final.weight", m.encoder.finalWeight, hidden), weights.input(b, "final.bias", m.encoder.finalBias, hidden)
	var taken []*tensor.Tensor
	m.encoder.cardBlocks(b, weights, program.embedded, tokens, func(index int, stream *tensor.Tensor) {
		for _, wanted := range m.spec.Layers {
			if wanted == index {
				normed := b.AffineLayerNorm(stream, finalWeight, finalBias, float32(g.Epsilon))
				// The patch tokens, a token's channels together, are the head's channel-first input over the grid.
				taken = append(taken, b.FlatSlice(normed, hidden, hidden, columns, rows))
			}
		}
	})
	head := depthHead{b: b, weights: weights}
	var level [4]*tensor.Tensor
	if len(taken) != len(level) {
		return nil, nil, errors.New("vitencoder depth card: the encoder gave other blocks than the head reads")
	}
	for index, x := range taken {
		x = head.conv(x, m.projects[index], fmt.Sprintf("projects.%d", index))
		if m.resize[index] != nil {
			x = head.conv(x, *m.resize[index], fmt.Sprintf("resize.%d", index))
		}
		level[index] = head.conv(x, m.levels[index], fmt.Sprintf("levels.%d", index))
	}
	if err := failed(); err != nil {
		return nil, nil, err
	}
	size := func(x *tensor.Tensor) (uint64, uint64) { return x.Shape.Dims[2], x.Shape.Dims[1] }
	path := head.fuse(level[3], nil, m.refine[3], "refine.3")(size(level[2]))
	for index := 2; index >= 0; index-- {
		if err := failed(); err != nil {
			return nil, nil, err
		}
		next := level[index]
		height, width := size(next)
		height, width = 2*height, 2*width
		if index > 0 {
			height, width = size(level[index-1])
		}
		path = head.fuse(path, next, m.refine[index], fmt.Sprintf("refine.%d", index))(height, width)
	}
	if err := failed(); err != nil {
		return nil, nil, err
	}
	out := head.resize(head.conv(path, m.output1, "output1"), uint64(height), uint64(width), "output1.resize")
	out = b.ReLU(head.conv(out, m.output2, "output2"))
	program.depth = b.ReLU(head.conv(out, m.output3, "output3"))
	return program, weights, failed()
}

// depthHead builds the DPT head's steps into b, their weights into weights, over channel-first activations, a pixel's
// channels together ([channels, width, height]).
type depthHead struct {
	b       *tensor.Builder
	weights cardWeights
}

// conv is the convolution c of x: a plain one as Conv2D, its weight [out][in][kernel][kernel] as the checkpoint holds
// it; a transposed one, whose kernel is its stride, as a 1 x 1 convolution to each output channel's kernel x kernel
// subpixels, subpixel by subpixel, which PixelShuffle2D spreads over the grid.
func (h depthHead) conv(x *tensor.Tensor, c depthConv, name string) *tensor.Tensor {
	in, out, kernel := uint64(c.in), uint64(c.out), uint64(c.kernel)
	if !c.transposed {
		var bias *tensor.Tensor
		if c.bias != nil {
			bias = h.weights.input(h.b, name+".bias", c.bias, out)
		}
		pad := uint32(c.padding)
		return h.b.Conv2D(x, h.weights.input(h.b, name+".weight", c.weight, kernel, kernel, in, out), bias,
			uint32(c.stride), uint32(c.stride), pad, pad, pad, pad, false)
	}
	area := c.kernel * c.kernel
	weight, bias := make([]float32, area*c.out*c.in), make([]float32, area*c.out)
	for i := range c.in {
		for o := range c.out {
			for ky := range c.kernel {
				for kx := range c.kernel {
					expanded := (ky*c.kernel+kx)*c.out + o
					weight[expanded*c.in+i] = c.weight[((i*c.out+o)*c.kernel+ky)*c.kernel+kx]
				}
			}
		}
	}
	for subpixel := range area {
		copy(bias[subpixel*c.out:], c.bias)
	}
	expanded := uint64(area) * out
	spread := h.b.Conv2D(x, h.weights.input(h.b, name+".weight", weight, 1, 1, in, expanded), h.weights.input(h.b, name+".bias", bias, expanded),
		1, 1, 0, 0, 0, 0, false)
	return h.b.PixelShuffle2D(spread, uint32(c.kernel))
}

// unit is a residual unit of x: x + conv2(relu(conv1(relu(x)))).
func (h depthHead) unit(x *tensor.Tensor, u residualUnit, name string) *tensor.Tensor {
	inner := h.conv(h.b.ReLU(x), u.conv1, name+".conv1")
	return h.b.Add(x, h.conv(h.b.ReLU(inner), u.conv2, name+".conv2"))
}

// fuse is the fusion block over path, plus the residual unit of level where it has one, through its second residual
// unit, resized to height by width and projected.
func (h depthHead) fuse(path, level *tensor.Tensor, block fusionBlock, name string) func(height, width uint64) *tensor.Tensor {
	return func(height, width uint64) *tensor.Tensor {
		if level != nil {
			path = h.b.Add(path, h.unit(level, block.unit1, name+".unit1"))
		}
		fused := h.unit(path, block.unit2, name+".unit2")
		return h.conv(h.resize(fused, height, width, name+".resize"), block.out, name+".out")
	}
}

// resize is x resized to height by width as torch's bilinear interpolation with corners aligned resizes it (the host's
// plane.resized): along the height, then the width, each a product with the interpolation's weights, the activation
// turned so the side it resizes comes first.
func (h depthHead) resize(x *tensor.Tensor, height, width uint64, name string) *tensor.Tensor {
	if x == nil {
		return nil
	}
	b := h.b
	channels, inWidth, inHeight := x.Shape.Dims[0], x.Shape.Dims[1], x.Shape.Dims[2]
	if inHeight != height {
		turned := b.Transpose2D(b.Reshape(x, channels*inWidth, inHeight))
		resized := b.MulMat(h.weights.input(b, name+".height", interpolation(inHeight, height), inHeight, height), turned)
		x = b.Reshape(b.Transpose2D(resized), channels, inWidth, height)
	}
	if inWidth != width {
		rowsFirst := b.Transpose2D(b.Reshape(x, channels*inWidth, height))
		turned := b.Transpose2D(b.Reshape(rowsFirst, height*channels, inWidth))
		resized := b.MulMat(h.weights.input(b, name+".width", interpolation(inWidth, width), inWidth, width), turned)
		back := b.Reshape(b.Transpose2D(resized), height, channels*width)
		x = b.Reshape(b.Transpose2D(back), channels, width, height)
	}
	return x
}

// interpolation is the bilinear interpolation, corners aligned, of in values to out as plane.resized weighs them: out
// rows of in weights, each output's two neighbours weighted by its fractions in float32.
func interpolation(in, out uint64) []float32 {
	weights := make([]float32, in*out)
	scale := float32(0)
	if out > 1 {
		scale = float32(in-1) / float32(out-1)
	}
	for o := range out {
		real := scale * float32(o)
		low := uint64(real)
		high := low
		if low < in-1 {
			high = low + 1
		}
		fraction := real - float32(low)
		weights[o*in+low] += 1 - fraction
		weights[o*in+high] += fraction
	}
	return weights
}
