package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

const hunyuanVLProjectorType = "hunyuanvl"

type HunyuanVLSpec struct {
	visionBackboneSpec
	OutputHidden     int
	MergeSize        int
	MinPixels        int
	MaxPixels        int
	ConvIntermediate int
	ProjectorInput   int
	PreLayerNorm     bool
	PostLayerNorm    bool
	FusedQKV         []bool
}

type HunyuanVLOutput gridOutput

type HunyuanVLRunner struct {
	projectorResources
	spec      HunyuanVLSpec
	attention visionAttentionPlan
}

func (r *HunyuanVLRunner) Spec() HunyuanVLSpec {
	if r == nil {
		return HunyuanVLSpec{}
	}
	return r.spec
}

func ReadHunyuanVLSpec(file *gguf.File) (HunyuanVLSpec, error) {
	spec := HunyuanVLSpec{}
	if err := readVisionBackbone(file, hunyuanVLProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return HunyuanVLSpec{}, err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{visionSpatialMergeKey, &spec.MergeSize},
		metadataIntField{visionMinPixelsKey, &spec.MinPixels},
		metadataIntField{visionMaxPixelsKey, &spec.MaxPixels},
	); err != nil {
		return HunyuanVLSpec{}, err
	}
	conv0, ok := file.Tensor("mm.0.weight")
	convIntermediate, conv0OK := checked.Int(conv0.Shape[tensor.TripleExtent])
	if !ok || conv0.Dimensions != tensor.MaxDimensions || !conv0OK {
		return HunyuanVLSpec{}, errors.New("projector: Hunyuan-VL first convolution is unavailable or invalid")
	}
	conv2, ok := file.Tensor("mm.2.weight")
	projectorInput, conv2OK := checked.Int(conv2.Shape[tensor.TripleExtent])
	if !ok || conv2.Dimensions != tensor.MaxDimensions || !conv2OK {
		return HunyuanVLSpec{}, errors.New("projector: Hunyuan-VL second convolution is unavailable or invalid")
	}
	spec.ConvIntermediate = convIntermediate
	spec.ProjectorInput = projectorInput
	spec.PreLayerNorm = hasTensor(file, visionPreNormWeightTensor)
	spec.PostLayerNorm = hasTensor(file, visionPostNormWeightTensor)
	spec.FusedQKV = make([]bool, spec.Layers)
	for layer := range spec.FusedQKV {
		spec.FusedQKV[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.attn_qkv.weight", layer))
	}
	if err := spec.validate(); err != nil {
		return HunyuanVLSpec{}, err
	}
	return spec, nil
}

func (s HunyuanVLSpec) validate() error {
	if err := s.visionBackboneSpec.validate(); err != nil {
		return err
	}
	if !checked.PositiveInts(s.OutputHidden, s.MergeSize, s.MinPixels, s.ConvIntermediate, s.ProjectorInput) ||
		s.MaxPixels < s.MinPixels || len(s.FusedQKV) != s.Layers {
		return fmt.Errorf("projector: invalid Hunyuan-VL metadata: %+v", s)
	}
	return nil
}

func validateHunyuanVLCatalog(file *gguf.File, spec HunyuanVLSpec) ([]string, error) {
	positionSide := spec.ImageSize / spec.PatchSize
	required := map[string][]uint64{
		"mm.pre_norm.weight":       {uint64(spec.Hidden)},
		"mm.0.weight":              {uint64(spec.MergeSize), uint64(spec.MergeSize), uint64(spec.Hidden), uint64(spec.ConvIntermediate)},
		"mm.0.bias":                {uint64(spec.ConvIntermediate)},
		"mm.2.weight":              {tensor.SingletonExtent, tensor.SingletonExtent, uint64(spec.ConvIntermediate), uint64(spec.ProjectorInput)},
		"mm.2.bias":                {uint64(spec.ProjectorInput)},
		visionImageNewlineTensor:   {uint64(spec.ProjectorInput)},
		multimodalProjectionWeight: {uint64(spec.ProjectorInput), uint64(spec.OutputHidden)},
		multimodalProjectionBias:   {uint64(spec.OutputHidden)},
		"mm.image_begin":           {uint64(spec.OutputHidden)}, "mm.image_end": {uint64(spec.OutputHidden)},
		"mm.post_norm.weight": {uint64(spec.OutputHidden)},
	}
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec, positionSide*positionSide, tensorOptional)
	if err := addOptionalVisionNormCatalog(file, required, spec.Hidden); err != nil {
		return nil, err
	}
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, spec.FusedQKV, false, tensorOptional)
	return validateProjectorTensorCatalog(file, required, "mm.0.weight")
}

func (r *HunyuanVLRunner) EncodeImage(ctx context.Context, source image.Image, options RasterPatchOptions) (HunyuanVLOutput, error) {
	spec := r.Spec()
	plan := spec.visionBackboneSpec.rasterPlan(spec.MergeSize, spec.MinPixels, spec.MaxPixels, rasterBicubic)
	return executePreparedProjector(
		ctx, r != nil && r.file != nil, source, plan, options, preprocessRasterPatches, r.encodeGraph,
	)
}
