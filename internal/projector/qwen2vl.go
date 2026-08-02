package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

const qwen2VLProjectorType = "qwen2vl_merger"

type Qwen2VLSpec struct {
	ImageSize          int
	PatchSize          int
	Hidden             int
	Intermediate       int
	MergerIntermediate int
	OutputHidden       int
	Layers             int
	Heads              int
	MergeSize          int
	LayerNormEpsilon   float32
	ImageMean          [3]float32
	ImageStd           [3]float32
	PreLayerNorm       bool
	PostLayerNorm      bool
	LegacyFFNSwapped   bool
}

type Qwen2VLOpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

type Qwen2VLRunner struct {
	file *gguf.File
	spec Qwen2VLSpec
	cuda *qwen3VLCUDA
}

type Qwen2VLImage = Qwen3VLImage
type Qwen2VLOutput = Qwen3VLOutput
type Qwen2VLPreprocessOptions = Qwen3VLPreprocessOptions

func OpenQwen2VL(path string) (*Qwen2VLRunner, error) {
	return OpenQwen2VLWithOptions(path, Qwen2VLOpenOptions{})
}

func OpenQwen2VLWithOptions(path string, options Qwen2VLOpenOptions) (*Qwen2VLRunner, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(openErr error) (*Qwen2VLRunner, error) {
		_ = file.Close()
		return nil, openErr
	}
	spec, err := ReadQwen2VLSpec(file)
	if err != nil {
		return fail(err)
	}
	if err := validateQwen2VLCatalog(file, spec); err != nil {
		return fail(err)
	}
	runner := &Qwen2VLRunner{file: file, spec: spec}
	if options.CUDA {
		runner.cuda, err = openQwen2VLCUDA(context.Background(), file, spec, options.DeviceOrdinal)
		if err != nil {
			return fail(fmt.Errorf("projector: initialize Qwen2-VL CUDA: %w", err))
		}
	}
	return runner, nil
}

func (r *Qwen2VLRunner) Close() error {
	if r == nil {
		return nil
	}
	var closeErr error
	if r.cuda != nil {
		closeErr = r.cuda.Close()
		r.cuda = nil
	}
	if r.file == nil {
		return closeErr
	}
	file := r.file
	r.file = nil
	return errors.Join(closeErr, file.Close())
}

func (r *Qwen2VLRunner) Spec() Qwen2VLSpec {
	if r == nil {
		return Qwen2VLSpec{}
	}
	return r.spec
}

func ReadQwen2VLSpec(file *gguf.File) (Qwen2VLSpec, error) {
	if file == nil {
		return Qwen2VLSpec{}, errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return Qwen2VLSpec{}, err
	}
	if architecture != "clip" {
		return Qwen2VLSpec{}, fmt.Errorf("projector: architecture %q is not clip", architecture)
	}
	projectorType, err := metadataString(file, "clip.projector_type")
	if err != nil {
		return Qwen2VLSpec{}, err
	}
	if projectorType != qwen2VLProjectorType {
		return Qwen2VLSpec{}, fmt.Errorf("projector: type %q is not %s", projectorType, qwen2VLProjectorType)
	}
	hasVision, err := metadataBool(file, "clip.has_vision_encoder")
	if err != nil {
		return Qwen2VLSpec{}, err
	}
	if !hasVision {
		return Qwen2VLSpec{}, errors.New("projector: vision encoder is disabled")
	}
	if useGELU, geluErr := metadataBool(file, "clip.use_gelu"); geluErr != nil {
		return Qwen2VLSpec{}, geluErr
	} else if !useGELU {
		return Qwen2VLSpec{}, errors.New("projector: Qwen2-VL GELU is disabled")
	}
	values := make([]int, 7)
	for index, key := range []string{
		"clip.vision.image_size", "clip.vision.patch_size", "clip.vision.embedding_length",
		"clip.vision.feed_forward_length", "clip.vision.projection_dim", "clip.vision.block_count",
		"clip.vision.attention.head_count",
	} {
		value, valueErr := metadataUint32(file, key)
		if valueErr != nil {
			return Qwen2VLSpec{}, valueErr
		}
		values[index] = int(value)
	}
	epsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return Qwen2VLSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return Qwen2VLSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
		return Qwen2VLSpec{}, err
	}
	merger, ok := file.Tensor("mm.0.weight")
	if !ok || merger.Dimensions != 2 || merger.Shape[1] > uint64(^uint(0)>>1) {
		return Qwen2VLSpec{}, errors.New("projector: merger input tensor is unavailable or invalid")
	}
	_, preWeight := file.Tensor("v.pre_ln.weight")
	_, preBias := file.Tensor("v.pre_ln.bias")
	_, postWeight := file.Tensor("v.post_ln.weight")
	_, postBias := file.Tensor("v.post_ln.bias")
	if preWeight != preBias || postWeight != postBias {
		return Qwen2VLSpec{}, errors.New("projector: Qwen2-VL optional norm tensors are incomplete")
	}
	mergeSize := 2
	if _, ok := file.MetadataValue("clip.vision.spatial_merge_size"); ok {
		value, valueErr := metadataUint32(file, "clip.vision.spatial_merge_size")
		if valueErr != nil {
			return Qwen2VLSpec{}, valueErr
		}
		mergeSize = int(value)
	}
	legacyFFN := false
	if down, ok := file.Tensor("v.blk.0.ffn_down.weight"); ok && down.Dimensions == 2 {
		legacyFFN = down.Shape[0] == uint64(values[2])
	}
	spec := Qwen2VLSpec{
		ImageSize: values[0], PatchSize: values[1], Hidden: values[2], Intermediate: values[3],
		OutputHidden: values[4], Layers: values[5], Heads: values[6], MergeSize: mergeSize,
		LayerNormEpsilon: epsilon, MergerIntermediate: int(merger.Shape[1]),
		PreLayerNorm: preWeight, PostLayerNorm: postWeight, LegacyFFNSwapped: legacyFFN,
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	if err := spec.validate(); err != nil {
		return Qwen2VLSpec{}, err
	}
	return spec, nil
}

func (s Qwen2VLSpec) validate() error {
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 ||
		s.MergerIntermediate <= 0 || s.OutputHidden <= 0 || s.Layers <= 0 || s.Heads <= 0 ||
		s.MergeSize != 2 || s.Hidden%s.Heads != 0 || (s.Hidden/s.Heads)%4 != 0 ||
		s.ImageSize%s.PatchSize != 0 || s.LayerNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid Qwen2-VL metadata: %+v", s)
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid normalization channel %d", channel)
		}
	}
	return nil
}

func (s Qwen2VLSpec) preprocessSpec() Qwen3VLSpec {
	return Qwen3VLSpec{
		ImageSize: s.ImageSize, PatchSize: s.PatchSize, Hidden: s.Hidden,
		Intermediate: s.Intermediate, MergerIntermediate: s.MergerIntermediate,
		OutputHidden: s.OutputHidden, Layers: s.Layers, Heads: s.Heads,
		MergeSize: s.MergeSize, LayerNormEpsilon: s.LayerNormEpsilon,
		ImageMean: s.ImageMean, ImageStd: s.ImageStd,
	}
}

func validateQwen2VLCatalog(file *gguf.File, spec Qwen2VLSpec) error {
	required := map[string][]uint64{
		"v.patch_embd.weight":   {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.patch_embd.weight.1": {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"mm.0.weight":           {uint64(spec.Hidden * 4), uint64(spec.MergerIntermediate)},
		"mm.0.bias":             {uint64(spec.MergerIntermediate)},
		"mm.2.weight":           {uint64(spec.MergerIntermediate), uint64(spec.OutputHidden)},
		"mm.2.bias":             {uint64(spec.OutputHidden)},
	}
	if spec.PreLayerNorm {
		required["v.pre_ln.weight"] = []uint64{uint64(spec.Hidden)}
		required["v.pre_ln.bias"] = []uint64{uint64(spec.Hidden)}
	}
	if spec.PostLayerNorm {
		required["v.post_ln.weight"] = []uint64{uint64(spec.Hidden)}
		required["v.post_ln.bias"] = []uint64{uint64(spec.Hidden)}
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		upShape := []uint64{uint64(spec.Hidden), uint64(spec.Intermediate)}
		downShape := []uint64{uint64(spec.Intermediate), uint64(spec.Hidden)}
		upBias := []uint64{uint64(spec.Intermediate)}
		downBias := []uint64{uint64(spec.Hidden)}
		if spec.LegacyFFNSwapped {
			upShape, downShape = downShape, upShape
			upBias, downBias = downBias, upBias
		}
		for name, shape := range map[string][]uint64{
			"attn_q.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_q.bias": {uint64(spec.Hidden)},
			"attn_k.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_k.bias": {uint64(spec.Hidden)},
			"attn_v.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_v.bias": {uint64(spec.Hidden)},
			"attn_out.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_out.bias": {uint64(spec.Hidden)},
			"ffn_up.weight": upShape, "ffn_up.bias": upBias,
			"ffn_down.weight": downShape, "ffn_down.bias": downBias,
			"ln1.weight": {uint64(spec.Hidden)}, "ln1.bias": {uint64(spec.Hidden)},
			"ln2.weight": {uint64(spec.Hidden)}, "ln2.bias": {uint64(spec.Hidden)},
		} {
			required[prefix+name] = shape
		}
	}
	for name, shape := range required {
		info, ok := file.Tensor(name)
		if !ok {
			return fmt.Errorf("projector: missing tensor %q", name)
		}
		if int(info.Dimensions) != len(shape) {
			return fmt.Errorf("projector: tensor %q rank %d, want %d", name, info.Dimensions, len(shape))
		}
		for dimension, want := range shape {
			if info.Shape[dimension] != want {
				return fmt.Errorf("projector: tensor %q shape %v, want %v", name, info.Shape[:info.Dimensions], shape)
			}
		}
	}
	return nil
}

func PreprocessQwen2VLImage(source image.Image, spec Qwen2VLSpec, options Qwen2VLPreprocessOptions) (Qwen2VLImage, error) {
	return PreprocessQwen3VLImage(source, spec.preprocessSpec(), options)
}

func PreprocessQwen2VLFrames(frames []image.Image, spec Qwen2VLSpec, options Qwen2VLPreprocessOptions) (Qwen2VLImage, error) {
	return PreprocessQwen3VLFrames(frames, spec.preprocessSpec(), options)
}

func (r *Qwen2VLRunner) EncodeImage(ctx context.Context, source image.Image, options Qwen2VLPreprocessOptions) (Qwen2VLOutput, error) {
	if r == nil || r.file == nil {
		return Qwen2VLOutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessQwen2VLImage(source, r.spec, options)
	if err != nil {
		return Qwen2VLOutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *Qwen2VLRunner) EncodeFrames(ctx context.Context, frames []image.Image, options Qwen2VLPreprocessOptions) (Qwen2VLOutput, error) {
	if r == nil || r.file == nil {
		return Qwen2VLOutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessQwen2VLFrames(frames, r.spec, options)
	if err != nil {
		return Qwen2VLOutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *Qwen2VLRunner) encode(ctx context.Context, input Qwen2VLImage) (Qwen2VLOutput, error) {
	if r.cuda != nil {
		return r.encodeCUDA(ctx, input)
	}
	hidden, err := r.patchEmbedding(ctx, input)
	if err != nil {
		return Qwen2VLOutput{}, err
	}
	rows := input.GridT * input.GridH * input.GridW
	rowOrder, columnOrder := mergedGrid(input.GridH, input.GridW, r.spec.MergeSize)
	if r.spec.PreLayerNorm {
		if hidden, err = r.normalize(ctx, hidden, rows, "v.pre_ln.weight", "v.pre_ln.bias"); err != nil {
			return Qwen2VLOutput{}, err
		}
	}
	for layer := 0; layer < r.spec.Layers; layer++ {
		if err := ctx.Err(); err != nil {
			return Qwen2VLOutput{}, err
		}
		if err := r.runLayer(ctx, hidden, rows, input.GridH*input.GridW, rowOrder, columnOrder, layer); err != nil {
			return Qwen2VLOutput{}, err
		}
	}
	if r.spec.PostLayerNorm {
		if hidden, err = r.normalize(ctx, hidden, rows, "v.post_ln.weight", "v.post_ln.bias"); err != nil {
			return Qwen2VLOutput{}, err
		}
	}
	embeddings, err := r.merge(ctx, hidden, rows)
	if err != nil {
		return Qwen2VLOutput{}, err
	}
	return Qwen2VLOutput{Embeddings: embeddings, GridT: input.GridT, GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}

func (r *Qwen2VLRunner) patchEmbedding(ctx context.Context, input Qwen2VLImage) ([]float32, error) {
	weight0, err := r.load(ctx, "v.patch_embd.weight")
	if err != nil {
		return nil, err
	}
	weight1, err := r.load(ctx, "v.patch_embd.weight.1")
	if err != nil {
		return nil, err
	}
	rows := input.GridT * input.GridH * input.GridW
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	temporalWidth := 3 * patchArea
	patchWidth := 2 * temporalWidth
	output := make([]float32, rows*r.spec.Hidden)
	parallelRows(rows, func(start, end int) {
		for row := start; row < end; row++ {
			source := input.PixelValues[row*patchWidth : (row+1)*patchWidth]
			for channel := 0; channel < r.spec.Hidden; channel++ {
				acc := 0.0
				base := channel * temporalWidth
				for color := 0; color < 3; color++ {
					for pixel := 0; pixel < patchArea; pixel++ {
						source0 := color*2*patchArea + pixel
						source1 := source0 + patchArea
						weight := base + color*patchArea + pixel
						acc += float64(source[source0])*float64(weight0.Data[weight]) + float64(source[source1])*float64(weight1.Data[weight])
					}
				}
				output[row*r.spec.Hidden+channel] = float32(acc)
			}
		}
	})
	return output, nil
}

func (r *Qwen2VLRunner) normalize(ctx context.Context, input []float32, rows int, weightName, biasName string) ([]float32, error) {
	weight, bias, err := r.loadPair(ctx, weightName, biasName)
	if err != nil {
		return nil, err
	}
	output := make([]float32, len(input))
	layerNorm(output, input, weight.Data, bias.Data, rows, r.spec.Hidden, r.spec.LayerNormEpsilon)
	return output, nil
}

func (r *Qwen2VLRunner) runLayer(ctx context.Context, hidden []float32, rows, temporalSpan int, rowOrder, columnOrder []int, layer int) error {
	prefix := fmt.Sprintf("v.blk.%d.", layer)
	normalized, err := r.normalize(ctx, hidden, rows, prefix+"ln1.weight", prefix+"ln1.bias")
	if err != nil {
		return err
	}
	qkv := make([]float32, rows*3*r.spec.Hidden)
	for projection, name := range []string{"attn_q", "attn_k", "attn_v"} {
		weight, bias, loadErr := r.loadPair(ctx, prefix+name+".weight", prefix+name+".bias")
		if loadErr != nil {
			return loadErr
		}
		values := linear(normalized, weight.Data, bias.Data, rows, r.spec.Hidden, r.spec.Hidden)
		for row := 0; row < rows; row++ {
			copy(qkv[row*3*r.spec.Hidden+projection*r.spec.Hidden:], values[row*r.spec.Hidden:(row+1)*r.spec.Hidden])
		}
	}
	attention := qwen3VLAttention(qkv, rows, temporalSpan, r.spec.preprocessSpec(), rowOrder, columnOrder)
	outWeight, outBias, err := r.loadPair(ctx, prefix+"attn_out.weight", prefix+"attn_out.bias")
	if err != nil {
		return err
	}
	projected := linear(attention, outWeight.Data, outBias.Data, rows, r.spec.Hidden, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += projected[index]
	}
	normalized, err = r.normalize(ctx, hidden, rows, prefix+"ln2.weight", prefix+"ln2.bias")
	if err != nil {
		return err
	}
	upName, downName := "ffn_up", "ffn_down"
	if r.spec.LegacyFFNSwapped {
		upName, downName = downName, upName
	}
	upWeight, upBias, err := r.loadPair(ctx, prefix+upName+".weight", prefix+upName+".bias")
	if err != nil {
		return err
	}
	up := linear(normalized, upWeight.Data, upBias.Data, rows, r.spec.Hidden, r.spec.Intermediate)
	for index, value := range up {
		up[index] = geluTanh(value)
	}
	downWeight, downBias, err := r.loadPair(ctx, prefix+downName+".weight", prefix+downName+".bias")
	if err != nil {
		return err
	}
	down := linear(up, downWeight.Data, downBias.Data, rows, r.spec.Intermediate, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += down[index]
	}
	return nil
}

func (r *Qwen2VLRunner) merge(ctx context.Context, hidden []float32, rows int) (reference.Value, error) {
	if rows%4 != 0 {
		return reference.Value{}, errors.New("projector: patch rows are not merge aligned")
	}
	mergedRows := rows / 4
	fc1Weight, fc1Bias, err := r.loadPair(ctx, "mm.0.weight", "mm.0.bias")
	if err != nil {
		return reference.Value{}, err
	}
	fc1 := linear(hidden, fc1Weight.Data, fc1Bias.Data, mergedRows, r.spec.Hidden*4, r.spec.MergerIntermediate)
	for index, value := range fc1 {
		fc1[index] = geluTanh(value)
	}
	fc2Weight, fc2Bias, err := r.loadPair(ctx, "mm.2.weight", "mm.2.bias")
	if err != nil {
		return reference.Value{}, err
	}
	output := linear(fc1, fc2Weight.Data, fc2Bias.Data, mergedRows, r.spec.MergerIntermediate, r.spec.OutputHidden)
	return reference.NewValue(tensor.MustShape(uint64(r.spec.OutputHidden), uint64(mergedRows)), output)
}

func (r *Qwen2VLRunner) load(ctx context.Context, name string) (reference.Value, error) {
	info, ok := r.file.Tensor(name)
	if !ok {
		return reference.Value{}, fmt.Errorf("projector: tensor %q is unavailable", name)
	}
	return model.LoadHostTensor(ctx, r.file, info)
}

func (r *Qwen2VLRunner) loadPair(ctx context.Context, first, second string) (reference.Value, reference.Value, error) {
	a, err := r.load(ctx, first)
	if err != nil {
		return reference.Value{}, reference.Value{}, err
	}
	b, err := r.load(ctx, second)
	return a, b, err
}
