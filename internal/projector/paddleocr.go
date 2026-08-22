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

const paddleOCRProjectorType = "paddleocr"

type PaddleOCRSpec struct {
	visionBackboneSpec
	ProjectorIntermediate int
	OutputHidden          int
	MergeSize             int
	MinPixels             int
	MaxPixels             int
	Activation            visionActivation
	PreLayerNorm          bool
	PostLayerNorm         bool
	FusedQKV              []bool
}

type PaddleOCROutput gridOutput

type PaddleOCRRunner struct {
	projectorResources
	spec      PaddleOCRSpec
	attention visionAttentionPlan
}

func (r *PaddleOCRRunner) Spec() PaddleOCRSpec {
	if r == nil {
		return PaddleOCRSpec{}
	}
	return r.spec
}

func ReadPaddleOCRSpec(file *gguf.File) (PaddleOCRSpec, error) {
	spec := PaddleOCRSpec{}
	if err := readRotaryVisionBackbone(file, paddleOCRProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return PaddleOCRSpec{}, err
	}
	if err := readProjectionNorm(file, &spec.ProjectionNormEpsilon); err != nil {
		return PaddleOCRSpec{}, err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{visionMinPixelsKey, &spec.MinPixels},
		metadataIntField{visionMaxPixelsKey, &spec.MaxPixels},
		metadataIntField{visionSpatialMergeKey, &spec.MergeSize},
	); err != nil {
		return PaddleOCRSpec{}, err
	}
	activation, err := readVisionActivationFlags(file)
	if err != nil {
		return PaddleOCRSpec{}, err
	}
	merger, ok := file.Tensor("mm.1.weight")
	spec.ProjectorIntermediate, ok = matrixRowsInt(merger, ok)
	if !ok {
		return PaddleOCRSpec{}, errors.New("projector: PaddleOCR merger tensor is unavailable or invalid")
	}
	spec.Activation = activation
	spec.PreLayerNorm = hasTensor(file, visionPreNormWeightTensor)
	spec.PostLayerNorm = hasTensor(file, visionPostNormWeightTensor)
	spec.FusedQKV = make([]bool, spec.Layers)
	for layer := range spec.FusedQKV {
		spec.FusedQKV[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.attn_qkv.weight", layer))
	}
	if err := spec.validate(); err != nil {
		return PaddleOCRSpec{}, err
	}
	return spec, nil
}

func readVisionActivationFlags(file *gguf.File) (visionActivation, error) {
	useGELU, err := optionalMetadataBool(file, "clip.use_gelu")
	if err != nil {
		return visionQuickGELU, err
	}
	useSiLU, err := optionalMetadataBool(file, "clip.use_silu")
	if err != nil {
		return visionQuickGELU, err
	}
	if useGELU && useSiLU {
		return visionQuickGELU, errors.New("projector: GELU and SiLU are both enabled")
	}
	if useGELU {
		return visionGELU, nil
	}
	if useSiLU {
		return visionSiLU, nil
	}
	return visionQuickGELU, nil
}

func optionalMetadataBool(file *gguf.File, key string) (bool, error) {
	value, ok := file.MetadataValue(key)
	if !ok {
		return false, nil
	}
	if value.Type != gguf.ValueTypeBool {
		return false, fmt.Errorf("projector: metadata %q must be bool", key)
	}
	result, ok := value.Data.(bool)
	if !ok {
		return false, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, nil
}

func hasTensor(file *gguf.File, name string) bool {
	_, ok := file.Tensor(name)
	return ok
}

func (s PaddleOCRSpec) validate() error {
	if err := s.visionBackboneSpec.validateRotary(); err != nil {
		return err
	}
	headWidth, headsOK := checked.DivExactInt(s.Hidden, s.Heads)
	_, rotaryOK := checked.DivExactInt(headWidth, tensor.PairedExtent*tensor.PairedExtent)
	if !checked.PositiveInts(s.ProjectorIntermediate, s.OutputHidden, s.MergeSize, s.MinPixels) ||
		s.MaxPixels < s.MinPixels || !checked.PositiveFinite32(s.ProjectionNormEpsilon) ||
		!headsOK || !rotaryOK || len(s.FusedQKV) != s.Layers {
		return fmt.Errorf("projector: invalid PaddleOCR metadata: %+v", s)
	}
	return nil
}

func validatePaddleOCRCatalog(file *gguf.File, spec PaddleOCRSpec) ([]string, error) {
	required := map[string][]uint64{
		"mm.input_norm.weight": {uint64(spec.Hidden)}, "mm.input_norm.bias": {uint64(spec.Hidden)},
		"mm.1.weight": {uint64(spec.Hidden * spec.MergeSize * spec.MergeSize), uint64(spec.ProjectorIntermediate)},
		"mm.1.bias":   {uint64(spec.ProjectorIntermediate)},
		"mm.2.weight": {uint64(spec.ProjectorIntermediate), uint64(spec.OutputHidden)},
		"mm.2.bias":   {uint64(spec.OutputHidden)},
	}
	positionSide := spec.ImageSize / spec.PatchSize
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec, positionSide*positionSide, tensorOptional)
	if err := addOptionalVisionNormCatalog(file, required, spec.Hidden); err != nil {
		return nil, err
	}
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, spec.FusedQKV, false, tensorOptional)
	return validateProjectorTensorCatalog(file, required)
}

func (r *PaddleOCRRunner) EncodeImage(ctx context.Context, source image.Image, options RasterPatchOptions) (PaddleOCROutput, error) {
	spec := r.Spec()
	plan := spec.visionBackboneSpec.rasterPlan(spec.MergeSize, spec.MinPixels, spec.MaxPixels, rasterBilinear)
	return executePreparedProjector(
		ctx, r != nil && r.file != nil, source, plan, options, preprocessRasterPatches, r.encodeGraph,
	)
}
