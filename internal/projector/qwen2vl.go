package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"llamacpp2go/internal/gguf"
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

type Qwen2VLOpenOptions = OpenOptions

type Qwen2VLRunner struct {
	file *gguf.File
	spec Qwen2VLSpec
	cuda *projectorCUDA
}

type Qwen2VLImage = Qwen3VLImage
type Qwen2VLOutput = Qwen3VLOutput
type Qwen2VLPreprocessOptions = Qwen3VLPreprocessOptions

func OpenQwen2VL(path string) (*Qwen2VLRunner, error) {
	return OpenQwen2VLWithOptions(path, Qwen2VLOpenOptions{})
}

func OpenQwen2VLWithOptions(path string, options Qwen2VLOpenOptions) (*Qwen2VLRunner, error) {
	return openCatalogProjector(path, options, "Qwen2-VL", nil,
		ReadQwen2VLSpec, validateQwen2VLCatalog,
		func(file *gguf.File, spec Qwen2VLSpec, cuda *projectorCUDA) *Qwen2VLRunner {
			return &Qwen2VLRunner{file: file, spec: spec, cuda: cuda}
		})
}

func (r *Qwen2VLRunner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
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
	spec := Qwen2VLSpec{}
	if err := readMetadataIntFields(file,
		metadataIntField{"clip.vision.image_size", &spec.ImageSize},
		metadataIntField{"clip.vision.patch_size", &spec.PatchSize},
		metadataIntField{"clip.vision.embedding_length", &spec.Hidden},
		metadataIntField{"clip.vision.feed_forward_length", &spec.Intermediate},
		metadataIntField{"clip.vision.projection_dim", &spec.OutputHidden},
		metadataIntField{"clip.vision.block_count", &spec.Layers},
		metadataIntField{"clip.vision.attention.head_count", &spec.Heads},
	); err != nil {
		return Qwen2VLSpec{}, err
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
		legacyFFN = down.Shape[0] == uint64(spec.Hidden)
	}
	spec.MergeSize = mergeSize
	spec.LayerNormEpsilon = epsilon
	spec.MergerIntermediate = int(merger.Shape[1])
	spec.PreLayerNorm = preWeight
	spec.PostLayerNorm = postWeight
	spec.LegacyFFNSwapped = legacyFFN
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

func validateQwen2VLCatalog(file *gguf.File, spec Qwen2VLSpec) ([]string, error) {
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
	return validateProjectorTensorCatalog(file, required)
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
	return r.encodeGraph(ctx, input)
}
