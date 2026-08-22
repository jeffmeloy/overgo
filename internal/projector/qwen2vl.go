package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
)

const qwen2VLProjectorType = "qwen2vl_merger"

type Qwen2VLSpec struct {
	visionBackboneSpec
	MergerIntermediate int
	OutputHidden       int
	MergeSize          int
	PreLayerNorm       bool
	PostLayerNorm      bool
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
	if err := readRotaryVisionBackbone(file, qwen2VLProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return Qwen2VLSpec{}, err
	}
	merger, ok := file.Tensor(projectionFirstWeightTensor)
	spec.MergerIntermediate, ok = matrixRowsInt(merger, ok)
	if !ok {
		return Qwen2VLSpec{}, errors.New("projector: merger input tensor is unavailable or invalid")
	}
	_, preWeight := file.Tensor(visionPreNormWeightTensor)
	_, preBias := file.Tensor(visionPreNormBiasTensor)
	_, postWeight := file.Tensor(visionPostNormWeightTensor)
	_, postBias := file.Tensor(visionPostNormBiasTensor)
	if preWeight != preBias || postWeight != postBias {
		return Qwen2VLSpec{}, errors.New("projector: Qwen2-VL optional norm tensors are incomplete")
	}
	mergeSize, err := metadataUint32(file, visionSpatialMergeKey)
	if err != nil {
		return Qwen2VLSpec{}, err
	}
	spec.MergeSize = int(mergeSize)
	spec.PreLayerNorm = preWeight
	spec.PostLayerNorm = postWeight
	if err := spec.validate(); err != nil {
		return Qwen2VLSpec{}, err
	}
	return spec, nil
}

func (s Qwen2VLSpec) validate() error {
	if err := s.visionBackboneSpec.validateRotary(); err != nil {
		return err
	}
	headWidth, headsOK := checked.DivExactInt(s.Hidden, s.Heads)
	_, rotaryOK := checked.DivExactInt(headWidth, tensor.PairedExtent*tensor.PairedExtent)
	if !checked.PositiveInts(s.MergerIntermediate, s.OutputHidden, s.MergeSize) || !headsOK || !rotaryOK {
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
		visionPatchWeightTensor:  {uint64(spec.PatchSize), uint64(spec.PatchSize), media.RGBChannels, uint64(spec.Hidden)},
		visionPatchWeightTensor1: {uint64(spec.PatchSize), uint64(spec.PatchSize), media.RGBChannels, uint64(spec.Hidden)},
	}
	addTwoLayerProjectionCatalog(file, required, spec.Hidden*spec.MergeSize*spec.MergeSize, spec.MergerIntermediate, spec.OutputHidden, tensorRequired)
	if spec.PreLayerNorm {
		required[visionPreNormWeightTensor] = []uint64{uint64(spec.Hidden)}
		required[visionPreNormBiasTensor] = []uint64{uint64(spec.Hidden)}
	}
	if spec.PostLayerNorm {
		required[visionPostNormWeightTensor] = []uint64{uint64(spec.Hidden)}
		required[visionPostNormBiasTensor] = []uint64{uint64(spec.Hidden)}
	}
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, nil, false, tensorRequired)
	return validateProjectorTensorCatalog(file, required)
}

func (r *Qwen2VLRunner) EncodeImage(ctx context.Context, source image.Image, options Qwen2VLPreprocessOptions) (Qwen2VLOutput, error) {
	return executePreparedProjector(ctx, r != nil && r.file != nil, func() (Qwen2VLImage, error) {
		return PreprocessQwen3VLImage(source, r.spec.preprocessSpec(), options)
	}, r.encodeGraph)
}

func (r *Qwen2VLRunner) EncodeFrames(ctx context.Context, frames []image.Image, options Qwen2VLPreprocessOptions) (Qwen2VLOutput, error) {
	return executePreparedProjector(ctx, r != nil && r.file != nil, func() (Qwen2VLImage, error) {
		return PreprocessQwen3VLFrames(frames, r.spec.preprocessSpec(), options)
	}, r.encodeGraph)
}
