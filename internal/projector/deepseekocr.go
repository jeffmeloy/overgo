package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tensorcatalog"
)

const (
	deepSeekOCRProjectorType = "deepseekocr"
	DeepSeekOCRImagePad      = "<image>"
)

type DeepSeekOCRSpec struct {
	ImageSize, TileSize, MinTiles, MaxTiles int
	PatchSize, Hidden, FeedForward, Layers  int
	Heads, SAMHidden, SAMLayers, SAMHeads   int
	Window, OutputHidden                    int
	LayerNormEpsilon                        float32
	ImageMean, ImageStd                     [media.RGBChannels]float32
	SAMGlobalLayers                         []bool
	TensorNames                             []string
}

type DeepSeekOCRInput struct {
	Tiles        []image.Image
	GridW, GridH int
}

type DeepSeekOCRRunner struct {
	projectorResources
	spec                      DeepSeekOCRSpec
	attention                 visionAttentionPlan
	samPosition, clipPosition reference.Value
	newline, separator        reference.Value
	dynamicTiles              bool
}

func openDeepSeekOCR(ctx context.Context, file *gguf.File, options OpenOptions) (*DeepSeekOCRRunner, error) {
	spec, err := ReadDeepSeekOCRSpec(file)
	if err != nil {
		return nil, err
	}
	runner := &DeepSeekOCRRunner{
		projectorResources: projectorResources{file: file}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads),
		dynamicTiles: !options.DisableDynamicTiles,
	}
	runner.samPosition, err = loadProjectorHostTensor(ctx, file, "v.sam.pos_embd.weight")
	if err != nil {
		return nil, err
	}
	runner.clipPosition, err = loadProjectorHostTensor(ctx, file, visionPositionWeightTensor)
	if err != nil {
		return nil, err
	}
	runner.newline, runner.separator, err = loadProjectorHostTensorPair(
		ctx, file, visionImageNewlineTensor, "v.view_seperator",
	)
	if err != nil {
		return nil, err
	}
	if err := runner.validateGraph(); err != nil {
		return nil, err
	}
	if options.CUDA {
		runner.cuda, err = openProjectorCUDA(ctx, file, spec.TensorNames, nil, options.DeviceOrdinal)
		if err != nil {
			return nil, fmt.Errorf("projector: initialize DeepSeek-OCR CUDA: %w", err)
		}
	}
	return runner, nil
}

func (r *DeepSeekOCRRunner) Spec() DeepSeekOCRSpec {
	if r == nil {
		return DeepSeekOCRSpec{}
	}
	return r.spec
}

func ReadDeepSeekOCRSpec(file *gguf.File) (DeepSeekOCRSpec, error) {
	spec, err := readDeepSeekOCRBaseSpec(file, deepSeekOCRProjectorType)
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	spec.TensorNames = deepSeekOCRTensorNames(file, spec)
	if err := validateDeepSeekOCRCatalog(file, spec); err != nil {
		return DeepSeekOCRSpec{}, err
	}
	return spec, nil
}

func readDeepSeekOCRBaseSpec(file *gguf.File, expectedType string) (DeepSeekOCRSpec, error) {
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
		"clip.vision.embedding_length":     &spec.Hidden,
		"clip.vision.block_count":          &spec.Layers,
		"clip.vision.attention.head_count": &spec.Heads,
		"clip.vision.sam.embedding_length": &spec.SAMHidden,
		"clip.vision.sam.block_count":      &spec.SAMLayers,
		"clip.vision.sam.head_count":       &spec.SAMHeads,
		"clip.vision.window_size":          &spec.Window,
		"clip.vision.preproc_image_size":   &spec.TileSize,
		"clip.vision.preproc_min_tiles":    &spec.MinTiles,
		"clip.vision.preproc_max_tiles":    &spec.MaxTiles,
	} {
		*target, err = read(key)
		if err != nil {
			return DeepSeekOCRSpec{}, err
		}
	}
	if err := deriveDeepSeekOCRDimensions(file, &spec); err != nil {
		return DeepSeekOCRSpec{}, err
	}
	spec.LayerNormEpsilon, err = metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", media.RGBChannels)
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", media.RGBChannels)
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	spec.SAMGlobalLayers, err = readMetadataBoolArray(file, "clip.vision.sam.global_attention_layers", true)
	if err != nil {
		return DeepSeekOCRSpec{}, err
	}
	projector, ok := file.Tensor(multimodalProjectionWeight)
	spec.OutputHidden, ok = matrixRowsInt(projector, ok)
	if !ok {
		return DeepSeekOCRSpec{}, errors.New("projector: DeepSeek-OCR projection tensor is unavailable or invalid")
	}
	if err := spec.validate(); err != nil {
		return DeepSeekOCRSpec{}, err
	}
	return spec, nil
}

func deriveDeepSeekOCRDimensions(file *gguf.File, spec *DeepSeekOCRSpec) error {
	patch, ok := file.Tensor("v.sam.patch_embd.weight")
	if !ok {
		return errors.New("projector: DeepSeek-OCR SAM patch tensor is unavailable")
	}
	if err := tensorcatalog.ValidateInfo(patch, tensorcatalog.Requirement{Rank: tensor.MaxDimensions, NonEmpty: true}); err != nil {
		return fmt.Errorf("projector: DeepSeek-OCR SAM patch tensor: %w", err)
	}
	if err := tensorcatalog.ValidateRelations(patch,
		[]tensorcatalog.EqualAxes{{tensor.FirstOffset, tensor.SingletonExtent}},
		[]tensorcatalog.FixedAxis{{Axis: tensor.PairedExtent, Extent: media.RGBChannels}, {Axis: tensor.TripleExtent, Extent: uint64(spec.SAMHidden)}},
	); err != nil {
		return fmt.Errorf("projector: DeepSeek-OCR SAM patch tensor: %w", err)
	}
	position, ok := file.Tensor("v.sam.pos_embd.weight")
	if !ok {
		return errors.New("projector: DeepSeek-OCR SAM position tensor is unavailable")
	}
	if err := tensorcatalog.ValidateInfo(position, tensorcatalog.Requirement{
		Ranks: []uint32{tensor.TripleExtent, tensor.MaxDimensions}, NonEmpty: true,
	}); err != nil {
		return fmt.Errorf("projector: DeepSeek-OCR SAM position tensor: %w", err)
	}
	if err := tensorcatalog.ValidateRelations(position,
		[]tensorcatalog.EqualAxes{{tensor.SingletonExtent, tensor.PairedExtent}},
		[]tensorcatalog.FixedAxis{
			{Axis: tensor.FirstOffset, Extent: uint64(spec.SAMHidden)},
			{Axis: tensor.TripleExtent, Extent: tensor.SingletonExtent, Optional: true},
		},
	); err != nil {
		return fmt.Errorf("projector: DeepSeek-OCR SAM position tensor: %w", err)
	}
	feedForward, ok := file.Tensor("v.blk.0.ffn_up.weight")
	if !ok {
		return errors.New("projector: DeepSeek-OCR feed-forward tensor is unavailable")
	}
	if err := tensorcatalog.ValidateInfo(feedForward, tensorcatalog.Requirement{Rank: tensor.PairedExtent, NonEmpty: true}); err != nil {
		return fmt.Errorf("projector: DeepSeek-OCR feed-forward tensor: %w", err)
	}
	if err := tensorcatalog.ValidateRelations(feedForward, nil,
		[]tensorcatalog.FixedAxis{{Axis: tensor.FirstOffset, Extent: uint64(spec.Hidden)}}); err != nil {
		return fmt.Errorf("projector: DeepSeek-OCR feed-forward tensor: %w", err)
	}
	spec.PatchSize = int(patch.Shape[tensor.FirstOffset])
	spec.ImageSize = int(position.Shape[tensor.SingletonExtent]) * spec.PatchSize
	spec.FeedForward = int(feedForward.Shape[tensor.SingletonExtent])
	return nil
}

func (s DeepSeekOCRSpec) validate() error {
	if s.ImageSize <= 0 || s.TileSize <= 0 || s.MinTiles <= 0 || s.MaxTiles < s.MinTiles ||
		s.PatchSize <= 0 || s.ImageSize%s.PatchSize != 0 || s.TileSize%s.PatchSize != 0 ||
		s.Hidden <= 0 || s.FeedForward <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.Hidden%s.Heads != 0 ||
		s.SAMHidden <= 0 || s.SAMLayers <= 0 || s.SAMHeads <= 0 || s.SAMHidden%s.SAMHeads != 0 ||
		s.Window <= 0 || s.OutputHidden <= 0 || s.LayerNormEpsilon <= 0 || len(s.SAMGlobalLayers) != s.SAMLayers {
		return fmt.Errorf("projector: invalid DeepSeek-OCR metadata: %+v", s)
	}
	for channel := range media.RGBChannels {
		if s.ImageStd[channel] <= 0 {
			return fmt.Errorf("projector: invalid DeepSeek-OCR image standard deviation %d", channel)
		}
	}
	return nil
}

func deepSeekOCRTensorNames(file *gguf.File, spec DeepSeekOCRSpec) []string {
	names := deepSeekOCRSAMTensorNames(spec)
	names = append(names,
		visionClassEmbeddingTensor, visionPositionWeightTensor,
		multimodalProjectionWeight, multimodalProjectionBias, visionImageNewlineTensor, "v.view_seperator",
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
	for _, name := range []string{visionPreNormWeightTensor, visionPreNormBiasTensor, visionPostNormWeightTensor, visionPostNormBiasTensor} {
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
		if err := tensorcatalog.ValidateInfo(info, tensorcatalog.Requirement{
			Ranks: []uint32{tensor.SingletonExtent, tensor.PairedExtent, tensor.TripleExtent, tensor.MaxDimensions},
		}); err != nil {
			return fmt.Errorf("projector: tensor %q rank %d is invalid", name, info.Dimensions)
		}
	}
	position, _ := file.Tensor("v.sam.pos_embd.weight")
	positionShape := []uint64{uint64(spec.SAMHidden), uint64(spec.ImageSize / spec.PatchSize), uint64(spec.ImageSize / spec.PatchSize)}
	if position.Dimensions == tensor.MaxDimensions {
		positionShape = append(positionShape, tensor.SingletonExtent)
	}
	requiredShapes := map[string][]uint64{
		"v.sam.pos_embd.weight":   positionShape,
		"v.sam.patch_embd.weight": {uint64(spec.PatchSize), uint64(spec.PatchSize), media.RGBChannels, uint64(spec.SAMHidden)},
		"v.sam.patch_embd.bias":   {uint64(spec.SAMHidden)},
		visionPositionWeightTensor: {uint64(spec.Hidden), uint64(
			(spec.ImageSize/spec.PatchSize/(tensor.PairedExtent*tensor.PairedExtent))*
				(spec.ImageSize/spec.PatchSize/(tensor.PairedExtent*tensor.PairedExtent)) + tensor.SingletonExtent,
		)},
		multimodalProjectionWeight: {uint64(tensor.PairedExtent * spec.Hidden), uint64(spec.OutputHidden)},
		multimodalProjectionBias:   {uint64(spec.OutputHidden)},
	}
	if err := validateProjectorTensorShapes(file, requiredShapes); err != nil {
		return err
	}
	for _, name := range []string{visionImageNewlineTensor, "v.view_seperator"} {
		info, _ := file.Tensor(name)
		elements, err := info.ElementCount()
		if err != nil {
			return err
		}
		if elements != uint64(spec.OutputHidden) {
			return fmt.Errorf("projector: tensor %q has %d elements, want %d", name, elements, spec.OutputHidden)
		}
	}
	for _, pair := range [][2]string{{visionPreNormWeightTensor, visionPreNormBiasTensor}, {visionPostNormWeightTensor, visionPostNormBiasTensor}} {
		_, first := file.Tensor(pair[0])
		_, second := file.Tensor(pair[1])
		if first != second {
			return fmt.Errorf("projector: tensors %q and %q must be paired", pair[0], pair[1])
		}
	}
	return nil
}

func preprocessDeepSeekOCRImage(source image.Image, spec DeepSeekOCRSpec, dynamicTiles bool) (DeepSeekOCRInput, error) {
	if source == nil {
		return DeepSeekOCRInput{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return DeepSeekOCRInput{}, err
	}
	bounds := source.Bounds()
	if bounds.Empty() {
		return DeepSeekOCRInput{}, errors.New("projector: image bounds are empty")
	}
	input := DeepSeekOCRInput{}
	if dynamicTiles && (bounds.Dx() > spec.TileSize || bounds.Dy() > spec.TileSize) {
		input.GridW, input.GridH = media.BestTileGrid(bounds.Dx(), bounds.Dy(), spec.TileSize, spec.MinTiles, spec.MaxTiles)
		refined := media.ResizeBicubic(source, input.GridW*spec.TileSize, input.GridH*spec.TileSize)
		for y := 0; y < input.GridH; y++ {
			for x := 0; x < input.GridW; x++ {
				input.Tiles = append(input.Tiles, cropImage(refined, image.Rect(
					x*spec.TileSize, y*spec.TileSize, (x+1)*spec.TileSize, (y+1)*spec.TileSize,
				)))
			}
		}
	}
	gray := color.Gray{Y: uint8(uint16(^uint8(0)) / tensor.PairedExtent)}
	input.Tiles = append(input.Tiles, resizeFitBicubic(source, spec.ImageSize, spec.ImageSize, gray))
	return input, nil
}

func (r *DeepSeekOCRRunner) EncodeImage(ctx context.Context, source image.Image) (reference.Value, error) {
	if r == nil || r.file == nil {
		return reference.Value{}, errRunnerClosed
	}
	input, err := preprocessDeepSeekOCRImage(source, r.spec, r.dynamicTiles)
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
	pixels := make([]float32, media.RGBChannels*size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			rawR, rawG, rawB, _ := source.At(source.Bounds().Min.X+x, source.Bounds().Min.Y+y).RGBA()
			for channel, raw := range [media.RGBChannels]uint32{rawR, rawG, rawB} {
				pixels[channel+media.RGBChannels*(x+size*y)] =
					(media.NormalizedRGBAChannel(raw) - r.spec.ImageMean[channel]) / r.spec.ImageStd[channel]
			}
		}
	}
	return pixels
}

func (r *DeepSeekOCRRunner) encodeTile(ctx context.Context, source image.Image) (reference.Value, error) {
	bounds := source.Bounds()
	size := bounds.Dx()
	_, aligned := checked.DivExactInt(size, r.spec.PatchSize)
	if bounds.Empty() || bounds.Dy() != size || !aligned {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR tile shape is invalid")
	}
	builder := tensor.NewBuilder()
	input := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(media.RGBChannels, uint64(size), uint64(size)))
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
	cur := convolveSame(builder, input, weight("v.sam.patch_embd.weight"), builder.Reshape(weight("v.sam.patch_embd.bias"), uint64(r.spec.SAMHidden)), uint32(r.spec.PatchSize), false)
	spatial := size / r.spec.PatchSize
	position := builder.Input("sam_position", dtype.F32, tensor.MustShape(uint64(r.spec.SAMHidden), uint64(spatial*spatial)))
	hostFeeds[position] = interpolateSpatialPosition(r.samPosition, r.spec.SAMHidden, spatial, spatial, false)
	cur = builder.Add(cur, builder.Reshape(position, uint64(r.spec.SAMHidden), uint64(spatial), uint64(spatial)))
	for layer := range r.spec.SAMLayers {
		prefix := fmt.Sprintf("v.sam.blk.%d.", layer)
		residual := cur
		norm := spatialAffineLayerNorm(
			builder, cur, weight(prefix+"pre_ln.weight"), weight(prefix+"pre_ln.bias"), r.spec.LayerNormEpsilon,
		)
		global := r.spec.SAMGlobalLayers[layer]
		width, height := uint32(norm.Shape.Dims[1]), uint32(norm.Shape.Dims[2])
		window := r.spec.Window
		if global {
			window = int(width)
			norm = builder.Reshape(norm, uint64(r.spec.SAMHidden), uint64(window*window), tensor.SingletonExtent)
		} else {
			norm = builder.WindowPartition2D(norm, uint32(window))
		}
		batches := norm.Shape.Dims[2]
		flat := builder.Reshape(norm, uint64(r.spec.SAMHidden), uint64(window*window)*batches)
		qkv := builder.MulMat(weight(prefix+"attn.qkv.weight"), flat)
		qkv = builder.Add(qkv, builder.Reshape(
			weight(prefix+"attn.qkv.bias"), uint64(tensor.TripleExtent*r.spec.SAMHidden), tensor.SingletonExtent,
		))
		headWidth := uint64(r.spec.SAMHidden / r.spec.SAMHeads)
		q := builder.GroupSlice(qkv, tensor.FirstOffset, headWidth, uint64(r.spec.SAMHeads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.SAMHidden), headWidth, uint64(r.spec.SAMHeads), headWidth)
		v := builder.GroupSlice(qkv, uint64(tensor.PairedExtent*r.spec.SAMHidden), headWidth, uint64(r.spec.SAMHeads), headWidth)
		shape := []uint64{headWidth, uint64(r.spec.SAMHeads), uint64(window * window), batches}
		q, k, v = builder.Reshape(q, shape...), builder.Reshape(k, shape...), builder.Reshape(v, shape...)
		attention := builder.SAMAttention(q, k, v, weight(prefix+"attn.pos_w.weight"), weight(prefix+"attn.pos_h.weight"),
			float32(1/math.Sqrt(float64(headWidth))), float32(tensor.SingletonExtent), uint32(window))
		attention = builder.Reshape(attention, uint64(r.spec.SAMHidden), uint64(window*window)*batches)
		attention = builder.MulMat(weight(prefix+"attn.out.weight"), attention)
		attention = builder.Add(attention, builder.Reshape(weight(prefix+"attn.out.bias"), uint64(r.spec.SAMHidden), tensor.SingletonExtent))
		attention = builder.Reshape(attention, uint64(r.spec.SAMHidden), uint64(window*window), batches)
		if global {
			attention = builder.Reshape(attention, uint64(r.spec.SAMHidden), uint64(width), uint64(height))
		} else {
			attention = builder.WindowUnpartition2D(attention, width, height)
		}
		cur = builder.Add(residual, attention)
		residual = cur
		norm = spatialAffineLayerNorm(
			builder, cur, weight(prefix+"post_ln.weight"), weight(prefix+"post_ln.bias"), r.spec.LayerNormEpsilon,
		)
		flat = builder.Reshape(norm, uint64(r.spec.SAMHidden), norm.Shape.Dims[1]*norm.Shape.Dims[2])
		upWeight := weight(prefix + "mlp.lin1.weight")
		up := builder.Add(builder.MulMat(upWeight, flat), builder.Reshape(weight(prefix+"mlp.lin1.bias"), upWeight.Shape.Dims[1], tensor.SingletonExtent))
		up = builder.GELU(up)
		down := builder.Add(builder.MulMat(weight(prefix+"mlp.lin2.weight"), up), builder.Reshape(weight(prefix+"mlp.lin2.bias"), uint64(r.spec.SAMHidden), tensor.SingletonExtent))
		cur = builder.Add(residual, builder.Reshape(down, residual.Shape.Dims[0], residual.Shape.Dims[1], residual.Shape.Dims[2]))
	}
	cur = convolveSame(builder, cur, weight("v.sam.neck.0.weight"), nil, tensor.SingletonExtent, false)
	cur = spatialAffineLayerNorm(
		builder, cur, weight("v.sam.neck.1.weight"), weight("v.sam.neck.1.bias"), r.spec.LayerNormEpsilon,
	)
	cur = convolveCentered(builder, cur, weight("v.sam.neck.2.weight"), nil, tensor.SingletonExtent, false)
	cur = spatialAffineLayerNorm(
		builder, cur, weight("v.sam.neck.3.weight"), weight("v.sam.neck.3.bias"), r.spec.LayerNormEpsilon,
	)
	cur = convolveCentered(builder, cur, weight("v.sam.net_2.weight"), nil, tensor.PairedExtent, false)
	cur = convolveCentered(builder, cur, weight("v.sam.net_3.weight"), nil, tensor.PairedExtent, false)
	return cur
}

func (r *DeepSeekOCRRunner) buildGraph(builder *tensor.Builder, input *tensor.Tensor, size int, weight func(string) *tensor.Tensor, hostFeeds map[*tensor.Tensor]reference.Value) *tensor.Tensor {
	cur := r.buildSAMGraph(builder, input, size, weight, hostFeeds)
	patches := cur.Shape.Dims[1] * cur.Shape.Dims[2]
	tokens := patches + tensor.SingletonExtent
	sam := builder.Reshape(cur, cur.Shape.Dims[0], patches)
	hidden := builder.Concat(builder.Reshape(weight(visionClassEmbeddingTensor), uint64(r.spec.Hidden), tensor.SingletonExtent), sam, tensor.SingletonExtent)
	clipPosition := builder.Input("clip_position", dtype.F32, tensor.MustShape(uint64(r.spec.Hidden), tokens))
	hostFeeds[clipPosition] = interpolateSpatialPosition(r.clipPosition, r.spec.Hidden, int(cur.Shape.Dims[1]), int(cur.Shape.Dims[2]), true)
	hidden = builder.Add(hidden, clipPosition)
	if hasTensor(r.file, visionPreNormWeightTensor) {
		hidden = builder.AffineLayerNorm(hidden, builder.Reshape(weight(visionPreNormWeightTensor), uint64(r.spec.Hidden)), builder.Reshape(weight(visionPreNormBiasTensor), uint64(r.spec.Hidden)), r.spec.LayerNormEpsilon)
	}
	for layer := range r.spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.AffineLayerNorm(hidden, builder.Reshape(weight(prefix+"ln1.weight"), uint64(r.spec.Hidden)), builder.Reshape(weight(prefix+"ln1.bias"), uint64(r.spec.Hidden)), r.spec.LayerNormEpsilon)
		qkv := builder.Add(
			builder.MulMat(weight(prefix+"attn_qkv.weight"), norm),
			builder.Reshape(
				weight(prefix+"attn_qkv.bias"), uint64(tensor.TripleExtent*r.spec.Hidden), tensor.SingletonExtent,
			),
		)
		headWidth := uint64(r.spec.Hidden / r.spec.Heads)
		q := builder.GroupSlice(qkv, tensor.FirstOffset, headWidth, uint64(r.spec.Heads), headWidth)
		k := builder.GroupSlice(qkv, uint64(r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		v := builder.GroupSlice(qkv, uint64(tensor.PairedExtent*r.spec.Hidden), headWidth, uint64(r.spec.Heads), headWidth)
		attention := r.attention.graph(builder, q, k, v)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), tokens)
		attention = builder.Add(builder.MulMat(weight(prefix+"attn_out.weight"), attention), builder.Reshape(weight(prefix+"attn_out.bias"), uint64(r.spec.Hidden), tensor.SingletonExtent))
		hidden = builder.Add(hidden, attention)
		norm = builder.AffineLayerNorm(hidden, builder.Reshape(weight(prefix+"ln2.weight"), uint64(r.spec.Hidden)), builder.Reshape(weight(prefix+"ln2.bias"), uint64(r.spec.Hidden)), r.spec.LayerNormEpsilon)
		up := builder.Add(builder.MulMat(weight(prefix+"ffn_up.weight"), norm), builder.Reshape(weight(prefix+"ffn_up.bias"), uint64(r.spec.FeedForward), tensor.SingletonExtent))
		up = builder.QuickGELU(up)
		down := builder.Add(builder.MulMat(weight(prefix+"ffn_down.weight"), up), builder.Reshape(weight(prefix+"ffn_down.bias"), uint64(r.spec.Hidden), tensor.SingletonExtent))
		hidden = builder.Add(hidden, down)
	}
	if hasTensor(r.file, visionPostNormWeightTensor) {
		hidden = builder.AffineLayerNorm(hidden, builder.Reshape(weight(visionPostNormWeightTensor), uint64(r.spec.Hidden)), builder.Reshape(weight(visionPostNormBiasTensor), uint64(r.spec.Hidden)), r.spec.LayerNormEpsilon)
	}
	clip := builder.FlatSlice(hidden, uint64(r.spec.Hidden), uint64(r.spec.Hidden), patches)
	joined := builder.Concat(clip, sam, tensor.FirstOffset)
	projected := builder.MulMat(weight(multimodalProjectionWeight), joined)
	return builder.Add(projected, builder.Reshape(weight(multimodalProjectionBias), uint64(r.spec.OutputHidden), tensor.SingletonExtent))
}

func spatialAffineLayerNorm(builder *tensor.Builder, input, weight, bias *tensor.Tensor, epsilon float32) *tensor.Tensor {
	channels, width, height := input.Shape.Dims[0], input.Shape.Dims[1], input.Shape.Dims[2]
	flat := builder.Reshape(input, channels, width*height)
	flat = builder.AffineLayerNorm(flat, builder.Reshape(weight, channels), builder.Reshape(bias, channels), epsilon)
	return builder.Reshape(flat, channels, width, height)
}

func interpolateSpatialPosition(source reference.Value, channels, width, height int, classLast bool) reference.Value {
	classTokens := tensor.FirstOffset
	spatialTokens := len(source.Data) / channels
	if classLast {
		classTokens = tensor.SingletonExtent
		spatialTokens -= classTokens
	}
	sourceSide := int(math.Sqrt(float64(spatialTokens)))
	output := make([]float32, channels*(width*height+classTokens))
	for y := range height {
		sy := (float64(y)+media.RasterSampleCenter)*float64(sourceSide)/float64(height) - media.RasterSampleCenter
		for x := range width {
			sx := (float64(x)+media.RasterSampleCenter)*float64(sourceSide)/float64(width) - media.RasterSampleCenter
			for channel := range channels {
				var sum, weights float64
				for oy := -tensor.SingletonExtent; oy <= tensor.PairedExtent; oy++ {
					py := max(tensor.FirstOffset, min(sourceSide-tensor.SingletonExtent, int(math.Floor(sy))+oy))
					wy := media.CubicConvolutionWeight(sy - float64(int(math.Floor(sy))+oy))
					for ox := -tensor.SingletonExtent; ox <= tensor.PairedExtent; ox++ {
						px := max(tensor.FirstOffset, min(sourceSide-tensor.SingletonExtent, int(math.Floor(sx))+ox))
						weight := wy * media.CubicConvolutionWeight(sx-float64(int(math.Floor(sx))+ox))
						sum += float64(source.Data[channel+channels*(px+sourceSide*py)]) * weight
						weights += weight
					}
				}
				output[channel+channels*(x+width*y)] = float32(sum / weights)
			}
		}
	}
	if classLast {
		copy(output[channels*width*height:], source.Data[channels*spatialTokens:channels*(spatialTokens+tensor.SingletonExtent)])
	}
	return reference.Value{Shape: tensor.MustShape(uint64(channels), uint64(width*height+classTokens)), Data: output}
}

func (r *DeepSeekOCRRunner) assemble(values []reference.Value, gridW, gridH int) (reference.Value, error) {
	if len(values) != gridW*gridH+tensor.SingletonExtent {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR tile grid is inconsistent")
	}
	hidden := r.spec.OutputHidden
	var output []float32
	appendToken := func(value reference.Value, token int) {
		output = append(output, value.Data[token*hidden:(token+tensor.SingletonExtent)*hidden]...)
	}
	if len(values) > tensor.SingletonExtent {
		patchSide, square := tensor.SquareSideInt(values[tensor.FirstOffset].Shape.Dims[1])
		if !square {
			return reference.Value{}, errors.New("projector: DeepSeek-OCR local tile output is not square")
		}
		for tileY := range gridH {
			for patchY := range patchSide {
				for tileX := range gridW {
					value := values[tileX+gridW*tileY]
					for patchX := range patchSide {
						appendToken(value, patchX+patchSide*patchY)
					}
				}
				output = append(output, r.newline.Data...)
			}
		}
	}
	overview := values[len(values)-1]
	patchSide, square := tensor.SquareSideInt(overview.Shape.Dims[1])
	if !square {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR overview output is not square")
	}
	for y := 0; y < patchSide; y++ {
		for x := 0; x < patchSide; x++ {
			appendToken(overview, x+patchSide*y)
		}
		output = append(output, r.newline.Data...)
	}
	output = append(output, r.separator.Data...)
	return reference.Value{Shape: tensor.MustShape(uint64(hidden), uint64(len(output)/hidden)), Data: output}, nil
}

func (r *DeepSeekOCRRunner) validateGraph() error {
	builder := tensor.NewBuilder()
	size := r.spec.ImageSize
	input := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(media.RGBChannels, uint64(size), uint64(size)))
	hostFeeds := make(map[*tensor.Tensor]reference.Value)
	weight := func(name string) *tensor.Tensor {
		info, _ := r.file.Tensor(name)
		return builder.Input(name, dtype.F32, tensor.MustShape(info.Extents()...))
	}
	_ = r.buildGraph(builder, input, size, weight, hostFeeds)
	if err := builder.Err(); err != nil {
		return fmt.Errorf("projector: validate DeepSeek-OCR graph: %w", err)
	}
	return nil
}

func (r *DeepSeekOCRRunner) imagesPrompt(ctx context.Context, tokenizer ImageTokenizer, sources []image.Image, text []string, _ PromptOptions) (MultimodalPrompt, error) {
	plan := delimitedImagePromptPlan("DeepSeek-OCR", DeepSeekOCRImagePad, "DeepSeek-OCR placeholder", true, r.spec.OutputHidden, "", "")
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, referenceImageEncoder(r.EncodeImage))
}
