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

type qwen2VLSpec struct {
	visionBackboneSpec
	MergerIntermediate int
	OutputHidden       int
	MergeSize          int
	PreLayerNorm       bool
	PostLayerNorm      bool
}

type qwen2VLRunner struct {
	projectorResources
	mediaPreprocessOwner
	spec qwen2VLSpec
}

type qwen2VLImage = Qwen3VLImage
type qwen2VLOutput = qwen3VLOutput
type qwen2VLPreprocessOptions = qwen3VLPreprocessOptions

func (r *qwen2VLRunner) Spec() qwen2VLSpec {
	if r == nil {
		return qwen2VLSpec{}
	}
	return r.spec
}

func readQwen2VLSpec(file *gguf.File) (qwen2VLSpec, error) {
	if useGELU, geluErr := metadataBool(file, visionUseGELUKey); geluErr != nil {
		return qwen2VLSpec{}, geluErr
	} else if !useGELU {
		return qwen2VLSpec{}, errors.New("projector: Qwen2-VL GELU is disabled")
	}
	spec := qwen2VLSpec{}
	if err := readRotaryVisionBackbone(file, qwen2VLProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return qwen2VLSpec{}, err
	}
	merger, ok := file.Tensor(projectionFirstWeightTensor)
	spec.MergerIntermediate, ok = matrixRowsInt(merger, ok)
	if !ok {
		return qwen2VLSpec{}, errors.New("projector: merger input tensor is unavailable or invalid")
	}
	_, preWeight := file.Tensor(visionPreNormWeightTensor)
	_, preBias := file.Tensor(visionPreNormBiasTensor)
	_, postWeight := file.Tensor(visionPostNormWeightTensor)
	_, postBias := file.Tensor(visionPostNormBiasTensor)
	if preWeight != preBias || postWeight != postBias {
		return qwen2VLSpec{}, errors.New("projector: Qwen2-VL optional norm tensors are incomplete")
	}
	mergeSize, err := metadataUint32(file, visionSpatialMergeKey)
	if err != nil {
		return qwen2VLSpec{}, err
	}
	spec.MergeSize = int(mergeSize)
	spec.PreLayerNorm = preWeight
	spec.PostLayerNorm = postWeight
	if err := spec.validate(); err != nil {
		return qwen2VLSpec{}, err
	}
	return spec, nil
}

func (s qwen2VLSpec) validate() error {
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

func (s qwen2VLSpec) preprocessSpec() qwen3VLSpec {
	return qwen3VLSpec{
		visionBackboneSpec: s.visionBackboneSpec,
		MergerIntermediate: s.MergerIntermediate, OutputHidden: s.OutputHidden, MergeSize: s.MergeSize,
	}
}

func validateQwen2VLCatalog(file *gguf.File, spec qwen2VLSpec) ([]string, error) {
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

func (r *qwen2VLRunner) EncodeImage(ctx context.Context, source image.Image, options qwen2VLPreprocessOptions) (qwen2VLOutput, error) {
	return r.EncodeFrames(ctx, []image.Image{source}, options)
}

func (r *qwen2VLRunner) EncodeFrames(ctx context.Context, frames []image.Image, options qwen2VLPreprocessOptions) (qwen2VLOutput, error) {
	return executePreparedProjector(
		ctx, r != nil && r.file != nil, frames, r.Spec().preprocessSpec(), options,
		PreprocessQwen3VLFrames, r.encodeGraph,
	)
}
