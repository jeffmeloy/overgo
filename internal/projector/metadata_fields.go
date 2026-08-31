package projector

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
)

func matrixRowsInt(info gguf.TensorInfo, found bool) (int, bool) {
	if !found || info.Dimensions != tensor.PairedExtent {
		return tensor.FirstOffset, false
	}
	return checked.Int(info.Shape[tensor.SingletonExtent])
}

func metadataInts(file *gguf.File, key string, presence tensorPresence, positive bool) ([]int, error) {
	value, ok := file.MetadataValue(key)
	if !ok {
		if presence == tensorRequired {
			return nil, fmt.Errorf("projector: metadata %q is unavailable", key)
		}
		return nil, nil
	}
	if value.Type != gguf.ValueTypeArray {
		return nil, fmt.Errorf("projector: metadata %q must be an integer array", key)
	}
	var result []int
	switch values := value.Data.(type) {
	case []int32:
		if value.ArrayType != gguf.ValueTypeInt32 {
			return nil, fmt.Errorf("projector: metadata %q has mismatched array type", key)
		}
		result = make([]int, len(values))
		for index, item := range values {
			result[index] = int(item)
		}
	case []uint32:
		if value.ArrayType != gguf.ValueTypeUint32 {
			return nil, fmt.Errorf("projector: metadata %q has mismatched array type", key)
		}
		result = make([]int, len(values))
		for index, item := range values {
			converted, valid := checked.Int(uint64(item))
			if !valid {
				return nil, fmt.Errorf("projector: metadata %q entry %d exceeds native limits", key, index)
			}
			result[index] = converted
		}
	default:
		return nil, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	if positive {
		for index, item := range result {
			if !checked.PositiveInts(item) {
				return nil, fmt.Errorf("projector: metadata %q entry %d is not positive", key, index)
			}
		}
	}
	return result, nil
}

const (
	visionImageSizeKey       = "clip.vision.image_size"
	visionPatchSizeKey       = "clip.vision.patch_size"
	visionHiddenKey          = "clip.vision.embedding_length"
	visionIntermediateKey    = "clip.vision.feed_forward_length"
	visionProjectionKey      = "clip.vision.projection_dim"
	visionLayerCountKey      = "clip.vision.block_count"
	visionHeadCountKey       = "clip.vision.attention.head_count"
	visionNormEpsilonKey     = "clip.vision.attention.layer_norm_epsilon"
	visionImageMeanKey       = "clip.vision.image_mean"
	visionImageStandardKey   = "clip.vision.image_std"
	visionProjectorTypeKey   = "clip.projector_type"
	visionTowerTypeKey       = "clip.vision.projector_type"
	visionEncoderEnabledKey  = "clip.has_vision_encoder"
	visionUseGELUKey         = "clip.use_gelu"
	visionUseSiLUKey         = "clip.use_silu"
	visionKVHeadCountKey     = "clip.vision.attention.head_count_kv"
	visionWindowSizeKey      = "clip.vision.window_size"
	audioProjectorTypeKey    = "clip.audio.projector_type"
	audioEncoderEnabledKey   = "clip.has_audio_encoder"
	visionSpatialMergeKey    = "clip.vision.spatial_merge_size"
	visionProjectorScaleKey  = "clip.vision.projector.scale_factor"
	visionRopeFrequencyKey   = "clip.vision.rope.freq_base"
	visionMinPixelsKey       = "clip.vision.image_min_pixels"
	visionMaxPixelsKey       = "clip.vision.image_max_pixels"
	visionMaxSoftTokensKey   = "clip.vision.max_soft_tokens"
	visionVideoSoftTokensKey = "clip.vision.video_max_soft_tokens"
	visionMaxGridSideKey     = "clip.vision.preproc_max_grid_side"
	visionProjectorNormKey   = "clip.vision.projector.layer_norm_epsilon"
	visionNormalizationWidth = media.RGBChannels
)

type visionBackboneSpec struct {
	ImageSize             int
	PatchSize             int
	Hidden                int
	Intermediate          int
	Layers                int
	Heads                 int
	LayerNormEpsilon      float32
	ProjectionNormEpsilon float32
	RopeFrequency         float32
	ImageMean             [visionNormalizationWidth]float32
	ImageStd              [visionNormalizationWidth]float32
}

func readProjectionNorm(file *gguf.File, target *float32) error {
	value, err := metadataFloat32(file, visionProjectorNormKey)
	if err == nil {
		*target = value
	}
	return err
}

func (s visionBackboneSpec) validate() error {
	_, headsOK := checked.DivExactInt(s.Hidden, s.Heads)
	_, patchesOK := checked.DivExactInt(s.ImageSize, s.PatchSize)
	if !checked.PositiveInts(s.ImageSize, s.PatchSize, s.Hidden, s.Intermediate, s.Layers, s.Heads) ||
		!headsOK || !patchesOK || !checked.PositiveFinite32(s.LayerNormEpsilon) {
		return fmt.Errorf("projector: invalid vision backbone metadata: %+v", s)
	}
	for channel := range s.ImageStd {
		if !checked.PositiveFinite32(s.ImageStd[channel]) || !checked.Finite32(s.ImageMean[channel]) {
			return fmt.Errorf("projector: invalid vision normalization channel %d", channel)
		}
	}
	return nil
}

func (s visionBackboneSpec) validateRotary() error {
	if err := s.validate(); err != nil {
		return err
	}
	if !checked.PositiveFinite32(s.RopeFrequency) {
		return fmt.Errorf("projector: invalid vision RoPE frequency %g", s.RopeFrequency)
	}
	return nil
}

func (s visionBackboneSpec) rasterPlan(mergeSize, minPixels, maxPixels int, interpolation rasterInterpolation) rasterPatchPlan {
	return rasterPatchPlan{
		patchSize: s.PatchSize, mergeSize: mergeSize,
		defaultBudget: pixelBudget{MinPixels: minPixels, MaxPixels: maxPixels},
		mean:          s.ImageMean, std: s.ImageStd, interpolation: interpolation,
	}
}

type metadataIntField struct {
	key    string
	target *int
}

func readMetadataIntFields(file *gguf.File, fields ...metadataIntField) error {
	for _, field := range fields {
		value, err := metadataUint32(file, field.key)
		if err != nil {
			return err
		}
		*field.target = int(value)
	}
	return nil
}

func readMetadataBoolArray(file *gguf.File, key string, required bool) ([]bool, error) {
	value, ok := file.MetadataValue(key)
	if !ok {
		if required {
			return nil, fmt.Errorf("projector: metadata %q is unavailable", key)
		}
		return nil, nil
	}
	values, storageOK := value.Data.([]bool)
	if value.Type != gguf.ValueTypeArray || value.ArrayType != gguf.ValueTypeBool || !storageOK {
		return nil, errors.New("projector: bool-array metadata is invalid")
	}
	return slices.Clone(values), nil
}

func readVisionBackbone(file *gguf.File, projectorType string, projection *int, spec *visionBackboneSpec) error {
	if err := validateVisionProjector(file, visionProjectorTypeKey, projectorType); err != nil {
		return err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{visionImageSizeKey, &spec.ImageSize},
		metadataIntField{visionPatchSizeKey, &spec.PatchSize},
		metadataIntField{visionHiddenKey, &spec.Hidden},
		metadataIntField{visionIntermediateKey, &spec.Intermediate},
		metadataIntField{visionProjectionKey, projection},
		metadataIntField{visionLayerCountKey, &spec.Layers},
		metadataIntField{visionHeadCountKey, &spec.Heads},
	); err != nil {
		return err
	}
	epsilon, err := metadataFloat32(file, visionNormEpsilonKey)
	if err != nil {
		return err
	}
	mean, err := metadataFloat32Array(file, visionImageMeanKey, len(spec.ImageMean))
	if err != nil {
		return err
	}
	standard, err := metadataFloat32Array(file, visionImageStandardKey, len(spec.ImageStd))
	if err != nil {
		return err
	}
	spec.LayerNormEpsilon = epsilon
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], standard)
	return nil
}

func readRotaryVisionBackbone(file *gguf.File, projectorType string, projection *int, spec *visionBackboneSpec) error {
	if err := readVisionBackbone(file, projectorType, projection, spec); err != nil {
		return err
	}
	// Converters may omit the vision rope base; the reference runtime
	// serves such towers at theta 10000, so an absent key takes that
	// declared-default while a present key of the wrong type still
	// refuses.
	if _, present := file.MetadataValue(visionRopeFrequencyKey); !present {
		spec.RopeFrequency = defaultVisionRopeFrequency
		return nil
	}
	frequency, err := metadataFloat32(file, visionRopeFrequencyKey)
	if err != nil {
		return err
	}
	spec.RopeFrequency = frequency
	return nil
}

// defaultVisionRopeFrequency is the reference runtime's rotary base for
// vision towers whose converter omitted the declaration.
const defaultVisionRopeFrequency = 10000
