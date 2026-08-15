package projector

import (
	"fmt"

	"overgo/internal/gguf"
)

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
	visionSpatialMergeKey    = "clip.vision.spatial_merge_size"
	visionProjectorScaleKey  = "clip.vision.projector.scale_factor"
	visionRopeFrequencyKey   = "clip.vision.rope.freq_base"
	visionMinPixelsKey       = "clip.vision.image_min_pixels"
	visionMaxPixelsKey       = "clip.vision.image_max_pixels"
	visionNormalizationWidth = rgbChannelCount
)

type visionBackboneSpec struct {
	ImageSize        int
	PatchSize        int
	Hidden           int
	Intermediate     int
	Layers           int
	Heads            int
	LayerNormEpsilon float32
	RopeFrequency    float32
	ImageMean        [visionNormalizationWidth]float32
	ImageStd         [visionNormalizationWidth]float32
}

func (s visionBackboneSpec) validate() error {
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 ||
		s.Layers <= 0 || s.Heads <= 0 || s.Hidden%s.Heads != 0 || s.ImageSize%s.PatchSize != 0 ||
		s.LayerNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid vision backbone metadata: %+v", s)
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid vision normalization channel %d", channel)
		}
	}
	return nil
}

func (s visionBackboneSpec) validateRotary() error {
	if err := s.validate(); err != nil {
		return err
	}
	if s.RopeFrequency <= 0 {
		return fmt.Errorf("projector: invalid vision RoPE frequency %g", s.RopeFrequency)
	}
	return nil
}

func (s visionBackboneSpec) rasterPlan(mergeSize, minPixels, maxPixels int, interpolation rasterInterpolation) rasterPatchPlan {
	return rasterPatchPlan{
		patchSize: s.PatchSize, mergeSize: mergeSize,
		defaultBudget: pixelBudget{MinPixels: minPixels, MaxPixels: maxPixels, MaxAspectRatio: defaultVisionMaxAspectRatio},
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
	frequency, err := metadataFloat32(file, visionRopeFrequencyKey)
	if err != nil {
		return err
	}
	spec.RopeFrequency = frequency
	return nil
}
