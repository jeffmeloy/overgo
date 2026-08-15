package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"overgo/internal/gguf"
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
	merge := 2
	if value, ok, valueErr := optionalMetadataUint32(file, "clip.vision.spatial_merge_size"); valueErr != nil {
		return HunyuanVLSpec{}, valueErr
	} else if ok {
		merge = int(value)
	}
	minPixels, _, err := optionalMetadataUint32(file, "clip.vision.image_min_pixels")
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	maxPixels, _, err := optionalMetadataUint32(file, "clip.vision.image_max_pixels")
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	factor := spec.PatchSize * merge
	if minPixels == 0 {
		minPixels = uint32(factor * factor * 256)
	}
	if maxPixels == 0 {
		maxPixels = uint32(factor * factor * 16384)
	}
	conv0, ok := file.Tensor("mm.0.weight")
	if !ok || conv0.Dimensions != 4 {
		return HunyuanVLSpec{}, errors.New("projector: Hunyuan-VL first convolution is unavailable or invalid")
	}
	conv2, ok := file.Tensor("mm.2.weight")
	if !ok || conv2.Dimensions != 4 {
		return HunyuanVLSpec{}, errors.New("projector: Hunyuan-VL second convolution is unavailable or invalid")
	}
	spec.MergeSize = merge
	spec.MinPixels = int(minPixels)
	spec.MaxPixels = int(maxPixels)
	spec.ConvIntermediate = int(conv0.Shape[3])
	spec.ProjectorInput = int(conv2.Shape[3])
	spec.PreLayerNorm = hasTensor(file, "v.pre_ln.weight")
	spec.PostLayerNorm = hasTensor(file, "v.post_ln.weight")
	spec.FusedQKV = make([]bool, spec.Layers)
	for layer := range spec.FusedQKV {
		spec.FusedQKV[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.attn_qkv.weight", layer))
	}
	if err := spec.validate(); err != nil {
		return HunyuanVLSpec{}, err
	}
	return spec, nil
}

func optionalMetadataUint32(file *gguf.File, key string) (uint32, bool, error) {
	value, ok := file.MetadataValue(key)
	if !ok {
		return 0, false, nil
	}
	if value.Type != gguf.ValueTypeUint32 {
		return 0, false, fmt.Errorf("projector: metadata %q must be uint32", key)
	}
	result, ok := value.Data.(uint32)
	if !ok {
		return 0, false, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, true, nil
}

func (s HunyuanVLSpec) validate() error {
	if err := s.visionBackboneSpec.validate(); err != nil {
		return err
	}
	if s.OutputHidden <= 0 || s.MergeSize <= 0 || s.MinPixels <= 0 || s.MaxPixels < s.MinPixels ||
		s.ConvIntermediate <= 0 || s.ProjectorInput <= 0 || len(s.FusedQKV) != s.Layers {
		return fmt.Errorf("projector: invalid Hunyuan-VL metadata: %+v", s)
	}
	return nil
}

func validateHunyuanVLCatalog(file *gguf.File, spec HunyuanVLSpec) ([]string, error) {
	positionSide := spec.ImageSize / spec.PatchSize
	required := map[string][]uint64{
		"v.patch_embd.weight":    {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.position_embd.weight": {uint64(spec.Hidden), uint64(positionSide * positionSide)},
		"mm.pre_norm.weight":     {uint64(spec.Hidden)},
		"mm.0.weight":            {uint64(spec.MergeSize), uint64(spec.MergeSize), uint64(spec.Hidden), uint64(spec.ConvIntermediate)},
		"mm.0.bias":              {uint64(spec.ConvIntermediate)},
		"mm.2.weight":            {1, 1, uint64(spec.ConvIntermediate), uint64(spec.ProjectorInput)},
		"mm.2.bias":              {uint64(spec.ProjectorInput)},
		"v.image_newline":        {uint64(spec.ProjectorInput)},
		"mm.model.fc.weight":     {uint64(spec.ProjectorInput), uint64(spec.OutputHidden)},
		"mm.model.fc.bias":       {uint64(spec.OutputHidden)},
		"mm.image_begin":         {uint64(spec.OutputHidden)}, "mm.image_end": {uint64(spec.OutputHidden)},
		"mm.post_norm.weight": {uint64(spec.OutputHidden)},
	}
	addOptionalProjectorTensor(file, required, "v.patch_embd.bias", []uint64{uint64(spec.Hidden)})
	for _, prefix := range []string{"v.pre_ln", "v.post_ln"} {
		if err := addOptionalProjectorPair(
			file, required, prefix+".weight", prefix+".bias", []uint64{uint64(spec.Hidden)},
		); err != nil {
			return nil, err
		}
	}
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, spec.FusedQKV)
	return validateProjectorTensorCatalog(file, required, "mm.0.weight")
}

func (r *HunyuanVLRunner) EncodeImage(ctx context.Context, source image.Image, options RasterPatchOptions) (HunyuanVLOutput, error) {
	if r == nil || r.file == nil {
		return HunyuanVLOutput{}, errRunnerClosed
	}
	plan := r.spec.visionBackboneSpec.rasterPlan(r.spec.MergeSize, r.spec.MinPixels, r.spec.MaxPixels, rasterBicubic)
	return encodeRasterPatches(ctx, source, options, plan, r.spec.validate, r.encodeGraph)
}
