package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

const (
	deepSeekOCRProjectorType      = "deepseekocr"
	deepSeekOCRSpatialNormEpsilon = 1e-6
	deepSeekOCRDefaultTileSize    = 640
	deepSeekOCRDefaultMaxTiles    = 9
	deepSeekOCRPaddingGray        = 127
	DeepSeekOCRImagePad           = "<image>"
)

type DeepSeekOCRSpec struct {
	ImageSize, TileSize, MinTiles, MaxTiles int
	PatchSize, Hidden, FeedForward, Layers  int
	Heads, SAMHidden, SAMLayers, SAMHeads   int
	Window, OutputHidden                    int
	LayerNormEpsilon                        float32
	ImageMean, ImageStd                     [3]float32
	TensorNames                             []string
}

type DeepSeekOCRInput struct {
	Tiles        []image.Image
	GridW, GridH int
}

type DeepSeekOCRRunner struct {
	file                      *gguf.File
	spec                      DeepSeekOCRSpec
	samPosition, clipPosition reference.Value
	cuda                      *projectorCUDA
}

type DeepSeekOCROpenOptions = OpenOptions

func OpenDeepSeekOCR(path string) (*DeepSeekOCRRunner, error) {
	return OpenDeepSeekOCRWithOptions(path, DeepSeekOCROpenOptions{})
}

func OpenDeepSeekOCRWithOptions(path string, options DeepSeekOCROpenOptions) (*DeepSeekOCRRunner, error) {
	return openProjectorResource(path, func(file *gguf.File) (*DeepSeekOCRRunner, error) {
		spec, err := ReadDeepSeekOCRSpec(file)
		if err != nil {
			return nil, err
		}
		runner := &DeepSeekOCRRunner{file: file, spec: spec}
		runner.samPosition, err = loadProjectorHostTensor(context.Background(), file, "v.sam.pos_embd.weight")
		if err != nil {
			return nil, err
		}
		runner.clipPosition, err = loadProjectorHostTensor(context.Background(), file, "v.position_embd.weight")
		if err != nil {
			return nil, err
		}
		if err := runner.validateGraph(); err != nil {
			return nil, err
		}
		if options.CUDA {
			runner.cuda, err = openProjectorCUDA(context.Background(), file, spec.TensorNames, nil, options.DeviceOrdinal)
			if err != nil {
				return nil, fmt.Errorf("projector: initialize DeepSeek-OCR CUDA: %w", err)
			}
		}
		return runner, nil
	})
}

func (r *DeepSeekOCRRunner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
}

func (r *DeepSeekOCRRunner) Spec() DeepSeekOCRSpec {
	if r == nil {
		return DeepSeekOCRSpec{}
	}
	return r.spec
}

func ReadDeepSeekOCRSpec(file *gguf.File) (DeepSeekOCRSpec, error) {
	spec, err := readDeepSeekOCRBaseSpec(
		file, deepSeekOCRProjectorType, deepSeekOCRDefaultTileSize, deepSeekOCRDefaultMaxTiles,
	)
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	spec.TensorNames = deepSeekOCRTensorNames(file, spec)
	if err := validateDeepSeekOCRCatalog(file, spec); err != nil {
		return DeepSeekOCRSpec{}, err
	}
	return spec, nil
}

func readDeepSeekOCRBaseSpec(file *gguf.File, expectedType string, tileSize, maxTiles int) (DeepSeekOCRSpec, error) {
	if err := validateVisionProjector(file, "clip.projector_type", expectedType); err != nil {
		return DeepSeekOCRSpec{}, err
	}
	var err error
	read := func(key string) (int, error) {
		value, valueErr := metadataUint32(file, key)
		return int(value), valueErr
	}
	var spec DeepSeekOCRSpec
	for key, target := range map[string]*int{
		"clip.vision.image_size":           &spec.ImageSize,
		"clip.vision.patch_size":           &spec.PatchSize,
		"clip.vision.embedding_length":     &spec.Hidden,
		"clip.vision.feed_forward_length":  &spec.FeedForward,
		"clip.vision.block_count":          &spec.Layers,
		"clip.vision.attention.head_count": &spec.Heads,
		"clip.vision.sam.embedding_length": &spec.SAMHidden,
		"clip.vision.sam.block_count":      &spec.SAMLayers,
		"clip.vision.sam.head_count":       &spec.SAMHeads,
		"clip.vision.window_size":          &spec.Window,
	} {
		*target, err = read(key)
		if err != nil {
			return DeepSeekOCRSpec{}, err
		}
	}
	spec.TileSize = tileSize
	if value, ok, valueErr := optionalMetadataUint32(file, "clip.vision.preproc_image_size"); valueErr != nil {
		return DeepSeekOCRSpec{}, valueErr
	} else if ok {
		spec.TileSize = int(value)
	}
	spec.MinTiles, spec.MaxTiles = 2, maxTiles
	if value, ok, valueErr := optionalMetadataUint32(file, "clip.vision.preproc_min_tiles"); valueErr != nil {
		return DeepSeekOCRSpec{}, valueErr
	} else if ok {
		spec.MinTiles = int(value)
	}
	if value, ok, valueErr := optionalMetadataUint32(file, "clip.vision.preproc_max_tiles"); valueErr != nil {
		return DeepSeekOCRSpec{}, valueErr
	} else if ok {
		spec.MaxTiles = int(value)
	}
	spec.LayerNormEpsilon, err = metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	projector, ok := file.Tensor("mm.model.fc.weight")
	if !ok || projector.Dimensions != 2 {
		return DeepSeekOCRSpec{}, errors.New("projector: DeepSeek-OCR projection tensor is unavailable or invalid")
	}
	spec.OutputHidden = int(projector.Shape[1])
	if err := spec.validate(); err != nil {
		return DeepSeekOCRSpec{}, err
	}
	return spec, nil
}

func (s DeepSeekOCRSpec) validate() error {
	if s.ImageSize <= 0 || s.TileSize <= 0 || s.MinTiles <= 0 || s.MaxTiles < s.MinTiles ||
		s.PatchSize <= 0 || s.ImageSize%s.PatchSize != 0 || s.TileSize%s.PatchSize != 0 ||
		s.Hidden <= 0 || s.FeedForward <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.Hidden%s.Heads != 0 ||
		s.SAMHidden <= 0 || s.SAMLayers <= 0 || s.SAMHeads <= 0 || s.SAMHidden%s.SAMHeads != 0 ||
		s.Window <= 0 || s.OutputHidden <= 0 || s.LayerNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid DeepSeek-OCR metadata: %+v", s)
	}
	for channel := range 3 {
		if s.ImageStd[channel] <= 0 {
			return fmt.Errorf("projector: invalid DeepSeek-OCR image standard deviation %d", channel)
		}
	}
	return nil
}

func deepSeekOCRTensorNames(file *gguf.File, spec DeepSeekOCRSpec) []string {
	names := deepSeekOCRSAMTensorNames(spec)
	names = append(names,
		"v.class_embd", "v.position_embd.weight",
		"mm.model.fc.weight", "mm.model.fc.bias", "v.image_newline", "v.view_seperator",
	)
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for _, suffix := range []string{
			"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "attn_qkv.weight", "attn_qkv.bias",
			"attn_out.weight", "attn_out.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias",
		} {
			names = append(names, prefix+suffix)
		}
	}
	for _, name := range []string{"v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias"} {
		if hasTensor(file, name) {
			names = append(names, name)
		}
	}
	return names
}

func deepSeekOCRSAMTensorNames(spec DeepSeekOCRSpec) []string {
	names := []string{
		"v.sam.pos_embd.weight", "v.sam.patch_embd.weight", "v.sam.patch_embd.bias",
		"v.sam.neck.0.weight", "v.sam.neck.1.weight", "v.sam.neck.1.bias",
		"v.sam.neck.2.weight", "v.sam.neck.3.weight", "v.sam.neck.3.bias",
		"v.sam.net_2.weight", "v.sam.net_3.weight",
	}
	for layer := 0; layer < spec.SAMLayers; layer++ {
		prefix := fmt.Sprintf("v.sam.blk.%d.", layer)
		for _, suffix := range []string{
			"pre_ln.weight", "pre_ln.bias", "post_ln.weight", "post_ln.bias",
			"attn.pos_h.weight", "attn.pos_w.weight", "attn.qkv.weight", "attn.qkv.bias",
			"attn.out.weight", "attn.out.bias", "mlp.lin1.weight", "mlp.lin1.bias",
			"mlp.lin2.weight", "mlp.lin2.bias",
		} {
			names = append(names, prefix+suffix)
		}
	}
	return names
}

func validateDeepSeekOCRCatalog(file *gguf.File, spec DeepSeekOCRSpec) error {
	for _, name := range spec.TensorNames {
		info, ok := file.Tensor(name)
		if !ok {
			return fmt.Errorf("projector: missing tensor %q", name)
		}
		if info.Dimensions == 0 || info.Dimensions > 4 {
			return fmt.Errorf("projector: tensor %q rank %d is invalid", name, info.Dimensions)
		}
	}
	requiredShapes := map[string][]uint64{
		"v.sam.pos_embd.weight":   {uint64(spec.SAMHidden), uint64(spec.ImageSize / spec.PatchSize), uint64(spec.ImageSize / spec.PatchSize)},
		"v.sam.patch_embd.weight": {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.SAMHidden)},
		"v.sam.patch_embd.bias":   {uint64(spec.SAMHidden)},
		"v.position_embd.weight":  {uint64(spec.Hidden), uint64((spec.ImageSize/spec.PatchSize/4)*(spec.ImageSize/spec.PatchSize/4) + 1)},
		"mm.model.fc.weight":      {uint64(2 * spec.Hidden), uint64(spec.OutputHidden)},
		"mm.model.fc.bias":        {uint64(spec.OutputHidden)},
	}
	if err := validateProjectorTensorShapes(file, requiredShapes); err != nil {
		return err
	}
	for _, name := range []string{"v.image_newline", "v.view_seperator"} {
		info, _ := file.Tensor(name)
		elements := uint64(1)
		for dimension := range info.Dimensions {
			elements *= info.Shape[dimension]
		}
		if elements != uint64(spec.OutputHidden) {
			return fmt.Errorf("projector: tensor %q has %d elements, want %d", name, elements, spec.OutputHidden)
		}
	}
	for _, pair := range [][2]string{{"v.pre_ln.weight", "v.pre_ln.bias"}, {"v.post_ln.weight", "v.post_ln.bias"}} {
		_, first := file.Tensor(pair[0])
		_, second := file.Tensor(pair[1])
		if first != second {
			return fmt.Errorf("projector: tensors %q and %q must be paired", pair[0], pair[1])
		}
	}
	return nil
}

func PreprocessDeepSeekOCRImage(source image.Image, spec DeepSeekOCRSpec) (DeepSeekOCRInput, error) {
	if source == nil {
		return DeepSeekOCRInput{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return DeepSeekOCRInput{}, err
	}
	bounds := source.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return DeepSeekOCRInput{}, errors.New("projector: image bounds are empty")
	}
	input := DeepSeekOCRInput{}
	if bounds.Dx() > spec.TileSize || bounds.Dy() > spec.TileSize {
		input.GridW, input.GridH = deepSeekOCRBestGrid(bounds.Dx(), bounds.Dy(), spec.TileSize, spec.MinTiles, spec.MaxTiles)
		refined := resizeImageBicubic(source, input.GridW*spec.TileSize, input.GridH*spec.TileSize)
		for y := 0; y < input.GridH; y++ {
			for x := 0; x < input.GridW; x++ {
				input.Tiles = append(input.Tiles, cropImage(refined, image.Rect(
					x*spec.TileSize, y*spec.TileSize, (x+1)*spec.TileSize, (y+1)*spec.TileSize,
				)))
			}
		}
	}
	input.Tiles = append(input.Tiles, deepSeekOCRFitPad(source, spec.ImageSize))
	return input, nil
}

func deepSeekOCRBestGrid(width, height, tile, minTiles, maxTiles int) (int, int) {
	aspect := float64(width) / float64(height)
	bestW, bestH, bestDiff := 1, 1, math.Inf(1)
	for count := minTiles; count <= maxTiles; count++ {
		for gridW := 1; gridW <= count; gridW++ {
			for gridH := 1; gridH <= count; gridH++ {
				if gridW*gridH < minTiles || gridW*gridH > maxTiles {
					continue
				}
				diff := math.Abs(aspect - float64(gridW)/float64(gridH))
				targetArea := tile * tile * gridW * gridH
				if diff < bestDiff || diff == bestDiff && width*height > targetArea/2 {
					bestW, bestH, bestDiff = gridW, gridH, diff
				}
			}
		}
	}
	return bestW, bestH
}

func deepSeekOCRFitPad(source image.Image, size int) image.Image {
	bounds := source.Bounds()
	scale := math.Min(float64(size)/float64(bounds.Dx()), float64(size)/float64(bounds.Dy()))
	width := max(1, min(size, int(math.Ceil(float64(bounds.Dx())*scale))))
	height := max(1, min(size, int(math.Ceil(float64(bounds.Dy())*scale))))
	resized := resizeImageBicubic(source, width, height)
	output := image.NewRGBA(image.Rect(0, 0, size, size))
	gray := color.RGBA{
		R: deepSeekOCRPaddingGray, G: deepSeekOCRPaddingGray, B: deepSeekOCRPaddingGray, A: opaqueAlpha,
	}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			output.SetRGBA(x, y, gray)
		}
	}
	offsetX, offsetY := (size-width)/2, (size-height)/2
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			output.Set(offsetX+x, offsetY+y, resized.At(x, y))
		}
	}
	return output
}

func (r *DeepSeekOCRRunner) EncodeImage(ctx context.Context, source image.Image) (reference.Value, error) {
	if r == nil || r.file == nil {
		return reference.Value{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessDeepSeekOCRImage(source, r.spec)
	if err != nil {
		return reference.Value{}, err
	}
	values := make([]reference.Value, len(input.Tiles))
	for index, tile := range input.Tiles {
		values[index], err = r.encodeTile(ctx, tile)
		if err != nil {
			return reference.Value{}, fmt.Errorf("projector: encode DeepSeek-OCR tile %d: %w", index, err)
		}
	}
	return r.assemble(values, input.GridW, input.GridH)
}

func (r *DeepSeekOCRRunner) tilePixels(source image.Image) []float32 {
	size := source.Bounds().Dx()
	pixels := make([]float32, rgbChannelCount*size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			rawR, rawG, rawB, _ := source.At(source.Bounds().Min.X+x, source.Bounds().Min.Y+y).RGBA()
			for channel, raw := range [rgbChannelCount]uint32{rawR, rawG, rawB} {
				pixels[channel+rgbChannelCount*(x+size*y)] =
					(normalizedImageChannel(raw) - r.spec.ImageMean[channel]) / r.spec.ImageStd[channel]
			}
		}
	}
	return pixels
}

func (r *DeepSeekOCRRunner) encodeTile(ctx context.Context, source image.Image) (reference.Value, error) {
	size := source.Bounds().Dx()
	if size <= 0 || source.Bounds().Dy() != size || size%r.spec.PatchSize != 0 {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR tile shape is invalid")
	}
	builder := tensor.NewBuilder()
	input := builder.Input("pixel_values", dtype.F32, tensor.MustShape(3, uint64(size), uint64(size)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[input] = pixelsValue(input, r.tilePixels(source))
	output := r.buildGraph(builder, input, size, graph.weight, graph.hostFeeds)
	results, err := graph.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *DeepSeekOCRRunner) buildSAMGraph(builder *tensor.Builder, input *tensor.Tensor, size int, weight func(string) *tensor.Tensor, hostFeeds map[*tensor.Tensor]reference.Value) *tensor.Tensor {
	cur := builder.Conv2D(input, weight("v.sam.patch_embd.weight"), builder.Reshape(weight("v.sam.patch_embd.bias"), uint64(r.spec.SAMHidden)),
		uint32(r.spec.PatchSize), uint32(r.spec.PatchSize), 0, 0, 0, 0, false)
	spatial := size / r.spec.PatchSize
	position := builder.Input("sam_position", dtype.F32, tensor.MustShape(uint64(r.spec.SAMHidden), uint64(spatial*spatial)))
	hostFeeds[position] = interpolateSpatialPosition(r.samPosition, r.spec.SAMHidden, spatial, spatial, false)
	cur = builder.Add(cur, builder.Reshape(position, uint64(r.spec.SAMHidden), uint64(spatial), uint64(spatial)))
	for layer := 0; layer < r.spec.SAMLayers; layer++ {
		prefix := fmt.Sprintf("v.sam.blk.%d.", layer)
		residual := cur
		norm := deepSeekOCRSpatialLayerNorm(builder, cur, weight(prefix+"pre_ln.weight"), weight(prefix+"pre_ln.bias"), deepSeekOCRSpatialNormEpsilon)
		global := layer == 2 || layer == 5 || layer == 8 || layer == 11
		width, height := uint32(norm.Shape.Dims[1]), uint32(norm.Shape.Dims[2])
		window := r.spec.Window
		if global {
			window = int(width)
			norm = builder.Reshape(norm, uint64(r.spec.SAMHidden), uint64(window*window), 1)
		} else {
			norm = builder.WindowPartition2D(norm, uint32(window))
		}
		batches := norm.Shape.Dims[2]
		flat := builder.Reshape(norm, uint64(r.spec.SAMHidden), uint64(window*window)*batches)
		qkv := builder.MulMat(weight(prefix+"attn.qkv.weight"), flat)
		qkv = builder.Add(qkv, builder.Reshape(weight(prefix+"attn.qkv.bias"), uint64(3*r.spec.SAMHidden), 1))
		headWidth := uint64(r.spec.SAMHidden / r.spec.SAMHeads)
		q := builder.GroupSlice(qkv, 0, headWidth, uint64(r.spec.SAMHeads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.SAMHidden), headWidth, uint64(r.spec.SAMHeads), headWidth)
		v := builder.GroupSlice(qkv, uint64(2*r.spec.SAMHidden), headWidth, uint64(r.spec.SAMHeads), headWidth)
		shape := []uint64{headWidth, uint64(r.spec.SAMHeads), uint64(window * window), batches}
		q, k, v = builder.Reshape(q, shape...), builder.Reshape(k, shape...), builder.Reshape(v, shape...)
		attention := builder.SAMAttention(q, k, v, weight(prefix+"attn.pos_w.weight"), weight(prefix+"attn.pos_h.weight"),
			float32(1/math.Sqrt(float64(headWidth))), 1, uint32(window))
		attention = builder.Reshape(attention, uint64(r.spec.SAMHidden), uint64(window*window)*batches)
		attention = builder.MulMat(weight(prefix+"attn.out.weight"), attention)
		attention = builder.Add(attention, builder.Reshape(weight(prefix+"attn.out.bias"), uint64(r.spec.SAMHidden), 1))
		attention = builder.Reshape(attention, uint64(r.spec.SAMHidden), uint64(window*window), batches)
		if global {
			attention = builder.Reshape(attention, uint64(r.spec.SAMHidden), uint64(width), uint64(height))
		} else {
			attention = builder.WindowUnpartition2D(attention, width, height)
		}
		cur = builder.Add(residual, attention)
		residual = cur
		norm = deepSeekOCRSpatialLayerNorm(builder, cur, weight(prefix+"post_ln.weight"), weight(prefix+"post_ln.bias"), deepSeekOCRSpatialNormEpsilon)
		flat = builder.Reshape(norm, uint64(r.spec.SAMHidden), norm.Shape.Dims[1]*norm.Shape.Dims[2])
		upWeight := weight(prefix + "mlp.lin1.weight")
		up := builder.Add(builder.MulMat(upWeight, flat), builder.Reshape(weight(prefix+"mlp.lin1.bias"), upWeight.Shape.Dims[1], 1))
		up = builder.GELU(up)
		down := builder.Add(builder.MulMat(weight(prefix+"mlp.lin2.weight"), up), builder.Reshape(weight(prefix+"mlp.lin2.bias"), uint64(r.spec.SAMHidden), 1))
		cur = builder.Add(residual, builder.Reshape(down, residual.Shape.Dims[0], residual.Shape.Dims[1], residual.Shape.Dims[2]))
	}
	cur = builder.Conv2D(cur, weight("v.sam.neck.0.weight"), nil, 1, 1, 0, 0, 0, 0, false)
	cur = deepSeekOCRSpatialLayerNorm(builder, cur, weight("v.sam.neck.1.weight"), weight("v.sam.neck.1.bias"), deepSeekOCRSpatialNormEpsilon)
	cur = builder.Conv2D(cur, weight("v.sam.neck.2.weight"), nil, 1, 1, 1, 1, 1, 1, false)
	cur = deepSeekOCRSpatialLayerNorm(builder, cur, weight("v.sam.neck.3.weight"), weight("v.sam.neck.3.bias"), deepSeekOCRSpatialNormEpsilon)
	cur = builder.Conv2D(cur, weight("v.sam.net_2.weight"), nil, 2, 2, 1, 1, 1, 1, false)
	cur = builder.Conv2D(cur, weight("v.sam.net_3.weight"), nil, 2, 2, 1, 1, 1, 1, false)
	return cur
}

func (r *DeepSeekOCRRunner) buildGraph(builder *tensor.Builder, input *tensor.Tensor, size int, weight func(string) *tensor.Tensor, hostFeeds map[*tensor.Tensor]reference.Value) *tensor.Tensor {
	cur := r.buildSAMGraph(builder, input, size, weight, hostFeeds)
	patches := cur.Shape.Dims[1] * cur.Shape.Dims[2]
	sam := builder.Reshape(cur, cur.Shape.Dims[0], patches)
	hidden := builder.Concat(builder.Reshape(weight("v.class_embd"), uint64(r.spec.Hidden), 1), sam, 1)
	clipPosition := builder.Input("clip_position", dtype.F32, tensor.MustShape(uint64(r.spec.Hidden), patches+1))
	hostFeeds[clipPosition] = interpolateSpatialPosition(r.clipPosition, r.spec.Hidden, int(cur.Shape.Dims[1]), int(cur.Shape.Dims[2]), true)
	hidden = builder.Add(hidden, clipPosition)
	if hasTensor(r.file, "v.pre_ln.weight") {
		hidden = builder.AffineLayerNorm(hidden, builder.Reshape(weight("v.pre_ln.weight"), uint64(r.spec.Hidden)), builder.Reshape(weight("v.pre_ln.bias"), uint64(r.spec.Hidden)), r.spec.LayerNormEpsilon)
	}
	for layer := 0; layer < r.spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.AffineLayerNorm(hidden, builder.Reshape(weight(prefix+"ln1.weight"), uint64(r.spec.Hidden)), builder.Reshape(weight(prefix+"ln1.bias"), uint64(r.spec.Hidden)), r.spec.LayerNormEpsilon)
		qkv := builder.Add(builder.MulMat(weight(prefix+"attn_qkv.weight"), norm), builder.Reshape(weight(prefix+"attn_qkv.bias"), uint64(3*r.spec.Hidden), 1))
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, 0, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(2*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		attention := builder.Attention(q, k, v, float32(1/math.Sqrt(float64(headWidth))), false)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), patches+1)
		attention = builder.Add(builder.MulMat(weight(prefix+"attn_out.weight"), attention), builder.Reshape(weight(prefix+"attn_out.bias"), uint64(r.spec.Hidden), 1))
		hidden = builder.Add(hidden, attention)
		norm = builder.AffineLayerNorm(hidden, builder.Reshape(weight(prefix+"ln2.weight"), uint64(r.spec.Hidden)), builder.Reshape(weight(prefix+"ln2.bias"), uint64(r.spec.Hidden)), r.spec.LayerNormEpsilon)
		up := builder.Add(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), builder.Reshape(weight(prefix+"ffn_up.bias"), uint64(r.spec.FeedForward), 1))
		up = builder.Multiply(up, builder.Sigmoid(builder.Scale(up, 1.702)))
		down := builder.Add(builder.MulMat(weight(prefix+"ffn_down.weight"), up), builder.Reshape(weight(prefix+"ffn_down.bias"), uint64(r.spec.Hidden), 1))
		hidden = builder.Add(hidden, down)
	}
	if hasTensor(r.file, "v.post_ln.weight") {
		hidden = builder.AffineLayerNorm(hidden, builder.Reshape(weight("v.post_ln.weight"), uint64(r.spec.Hidden)), builder.Reshape(weight("v.post_ln.bias"), uint64(r.spec.Hidden)), r.spec.LayerNormEpsilon)
	}
	clip := builder.FlatSlice(hidden, uint64(r.spec.Hidden), uint64(r.spec.Hidden), patches)
	joined := builder.Concat(clip, sam, 0)
	projected := builder.MulMat(weight("mm.model.fc.weight"), joined)
	return builder.Add(projected, builder.Reshape(weight("mm.model.fc.bias"), uint64(r.spec.OutputHidden), 1))
}

func deepSeekOCRSpatialLayerNorm(builder *tensor.Builder, input, weight, bias *tensor.Tensor, epsilon float32) *tensor.Tensor {
	channels, width, height := input.Shape.Dims[0], input.Shape.Dims[1], input.Shape.Dims[2]
	flat := builder.Reshape(input, channels, width*height)
	flat = builder.AffineLayerNorm(flat, builder.Reshape(weight, channels), builder.Reshape(bias, channels), epsilon)
	return builder.Reshape(flat, channels, width, height)
}

func cubicWeight(value float64) float64 {
	value = math.Abs(value)
	const alpha = -0.75
	if value <= 1 {
		return ((alpha+2)*value-(alpha+3))*value*value + 1
	}
	if value < 2 {
		return ((alpha*value-5*alpha)*value+8*alpha)*value - 4*alpha
	}
	return 0
}

func interpolateSpatialPosition(source reference.Value, channels, width, height int, classLast bool) reference.Value {
	classTokens := 0
	spatialTokens := int(source.Shape.Dims[1])
	if source.Shape.Rank == 3 {
		spatialTokens *= int(source.Shape.Dims[2])
	}
	if classLast {
		classTokens = 1
		spatialTokens--
	}
	sourceSide := int(math.Sqrt(float64(spatialTokens)))
	output := make([]float32, channels*(width*height+classTokens))
	for y := 0; y < height; y++ {
		sy := (float64(y)+0.5)*float64(sourceSide)/float64(height) - 0.5
		for x := 0; x < width; x++ {
			sx := (float64(x)+0.5)*float64(sourceSide)/float64(width) - 0.5
			for channel := 0; channel < channels; channel++ {
				var sum, weights float64
				for oy := -1; oy <= 2; oy++ {
					py := max(0, min(sourceSide-1, int(math.Floor(sy))+oy))
					wy := cubicWeight(sy - float64(int(math.Floor(sy))+oy))
					for ox := -1; ox <= 2; ox++ {
						px := max(0, min(sourceSide-1, int(math.Floor(sx))+ox))
						weight := wy * cubicWeight(sx-float64(int(math.Floor(sx))+ox))
						sum += float64(source.Data[channel+channels*(px+sourceSide*py)]) * weight
						weights += weight
					}
				}
				output[channel+channels*(x+width*y)] = float32(sum / weights)
			}
		}
	}
	if classLast {
		copy(output[channels*width*height:], source.Data[channels*spatialTokens:channels*(spatialTokens+1)])
	}
	return reference.Value{Shape: tensor.MustShape(uint64(channels), uint64(width*height+classTokens)), Data: output}
}

func (r *DeepSeekOCRRunner) assemble(values []reference.Value, gridW, gridH int) (reference.Value, error) {
	if len(values) != gridW*gridH+1 {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR tile grid is inconsistent")
	}
	newline, err := loadProjectorHostTensor(context.Background(), r.file, "v.image_newline")
	if err != nil {
		return reference.Value{}, err
	}
	separator, err := loadProjectorHostTensor(context.Background(), r.file, "v.view_seperator")
	if err != nil {
		return reference.Value{}, err
	}
	hidden := r.spec.OutputHidden
	var output []float32
	appendToken := func(value reference.Value, token int) {
		output = append(output, value.Data[token*hidden:(token+1)*hidden]...)
	}
	if gridW > 0 {
		patchSide := int(math.Sqrt(float64(values[0].Shape.Dims[1])))
		if patchSide*patchSide != int(values[0].Shape.Dims[1]) {
			return reference.Value{}, errors.New("projector: DeepSeek-OCR local tile output is not square")
		}
		for tileY := 0; tileY < gridH; tileY++ {
			for patchY := 0; patchY < patchSide; patchY++ {
				for tileX := 0; tileX < gridW; tileX++ {
					value := values[tileX+gridW*tileY]
					for patchX := 0; patchX < patchSide; patchX++ {
						appendToken(value, patchX+patchSide*patchY)
					}
				}
				output = append(output, newline.Data...)
			}
		}
	}
	overview := values[len(values)-1]
	patchSide := int(math.Sqrt(float64(overview.Shape.Dims[1])))
	if patchSide*patchSide != int(overview.Shape.Dims[1]) {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR overview output is not square")
	}
	for y := 0; y < patchSide; y++ {
		for x := 0; x < patchSide; x++ {
			appendToken(overview, x+patchSide*y)
		}
		output = append(output, newline.Data...)
	}
	output = append(output, separator.Data...)
	return reference.Value{Shape: tensor.MustShape(uint64(hidden), uint64(len(output)/hidden)), Data: output}, nil
}

func (r *DeepSeekOCRRunner) validateGraph() error {
	builder := tensor.NewBuilder()
	size := r.spec.ImageSize
	input := builder.Input("pixel_values", dtype.F32, tensor.MustShape(3, uint64(size), uint64(size)))
	hostFeeds := make(map[*tensor.Tensor]reference.Value)
	weight := func(name string) *tensor.Tensor {
		info, _ := r.file.Tensor(name)
		return builder.Input(name, dtype.F32, tensorInfoShape(info))
	}
	_ = r.buildGraph(builder, input, size, weight, hostFeeds)
	if err := builder.Err(); err != nil {
		return fmt.Errorf("projector: validate DeepSeek-OCR graph: %w", err)
	}
	return nil
}

func (r *DeepSeekOCRRunner) BuildImagePrompt(ctx context.Context, tokenizer ImageTokenizer, source image.Image, beforeImage, afterImage string, _ bool) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *DeepSeekOCRRunner) BuildImagesPrompt(ctx context.Context, tokenizer ImageTokenizer, sources []image.Image, text []string, _ bool) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, false)
}

func (r *DeepSeekOCRRunner) BuildImagesHistoryPrompt(ctx context.Context, tokenizer ImageTokenizer, sources []image.Image, text []string) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *DeepSeekOCRRunner) buildImagesPrompt(ctx context.Context, tokenizer ImageTokenizer, sources []image.Image, text []string, history bool) (MultimodalPrompt, error) {
	return executeImagePromptPlan(ctx, tokenizer, sources, text, imagePromptPlan{
		Family: "DeepSeek-OCR", Placeholder: DeepSeekOCRImagePad, PlaceholderLabel: "DeepSeek-OCR placeholder",
		History: history, EmbeddingWidth: r.spec.OutputHidden,
		Render: func(text []string, items []imagePromptItem) string {
			return renderDelimitedImagePrompt(text, items, DeepSeekOCRImagePad, "", "\n")
		},
	}, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		value, err := r.EncodeImage(ctx, source)
		return imagePromptItem{Embeddings: value.Data, Count: int(value.Shape.Dims[1])}, err
	})
}
