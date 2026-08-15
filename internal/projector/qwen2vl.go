package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"overgo/internal/gguf"
)

const qwen2VLProjectorType = "qwen2vl_merger"

type Qwen2VLSpec struct {
	visionBackboneSpec
	MergerIntermediate int
	OutputHidden       int
	MergeSize          int
	PreLayerNorm       bool
	PostLayerNorm      bool
	LegacyFFNSwapped   bool
}

type Qwen2VLRunner struct {
	projectorResources
	spec Qwen2VLSpec
}

type Qwen2VLImage = Qwen3VLImage
type Qwen2VLOutput = Qwen3VLOutput
type Qwen2VLPreprocessOptions = Qwen3VLPreprocessOptions

func (r *Qwen2VLRunner) Spec() Qwen2VLSpec {
	if r == nil {
		return Qwen2VLSpec{}
	}
	return r.spec
}

func ReadQwen2VLSpec(file *gguf.File) (Qwen2VLSpec, error) {
	if useGELU, geluErr := metadataBool(file, "clip.use_gelu"); geluErr != nil {
		return Qwen2VLSpec{}, geluErr
	} else if !useGELU {
		return Qwen2VLSpec{}, errors.New("projector: Qwen2-VL GELU is disabled")
	}
	spec := Qwen2VLSpec{}
	if err := readVisionBackbone(file, qwen2VLProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
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
	spec.MergerIntermediate = int(merger.Shape[1])
	spec.PreLayerNorm = preWeight
	spec.PostLayerNorm = postWeight
	spec.LegacyFFNSwapped = legacyFFN
	if err := spec.validate(); err != nil {
		return Qwen2VLSpec{}, err
	}
	return spec, nil
}

func (s Qwen2VLSpec) validate() error {
	if err := s.visionBackboneSpec.validate(); err != nil {
		return err
	}
	if s.MergerIntermediate <= 0 || s.OutputHidden <= 0 || s.MergeSize != 2 || (s.Hidden/s.Heads)%4 != 0 {
		return fmt.Errorf("projector: invalid Qwen2-VL metadata: %+v", s)
	}
	return nil
}

func (s Qwen2VLSpec) preprocessSpec() Qwen3VLSpec {
	return Qwen3VLSpec{
		visionBackboneSpec: s.visionBackboneSpec,
		MergerIntermediate: s.MergerIntermediate, OutputHidden: s.OutputHidden, MergeSize: s.MergeSize,
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
		return Qwen2VLOutput{}, errRunnerClosed
	}
	input, err := PreprocessQwen2VLImage(source, r.spec, options)
	if err != nil {
		return Qwen2VLOutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *Qwen2VLRunner) EncodeFrames(ctx context.Context, frames []image.Image, options Qwen2VLPreprocessOptions) (Qwen2VLOutput, error) {
	if r == nil || r.file == nil {
		return Qwen2VLOutput{}, errRunnerClosed
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
