package media

import (
	"errors"
	"math"
)

// PairedShiftScaleGateFields resolves the field count for two modulation
// triplets, each containing shift, scale, and residual gate vectors.
func PairedShiftScaleGateFields(configured int) int {
	if configured > 0 {
		return configured
	}
	return 2 * 3
}

func DefaultPairedShiftScaleGateFields() int { return PairedShiftScaleGateFields(0) }

// PairedShiftScaleGateWidth returns the corresponding flattened width.
func PairedShiftScaleGateWidth(hidden int) int { return DefaultPairedShiftScaleGateFields() * hidden }

// ShiftScaleGateOffsets names the flattened paired-triplet field order.
type ShiftScaleGateOffsets struct {
	PreScale, PreShift, PreGate    int
	PostScale, PostShift, PostGate int
}

// PairedShiftScaleGateOffsets returns the neutral paired-triplet layout.
func PairedShiftScaleGateOffsets() ShiftScaleGateOffsets {
	return ShiftScaleGateOffsets{PreScale: 0, PreShift: 1, PreGate: 2, PostScale: 3, PostShift: 4, PostGate: 5}
}

// ShiftFirstGateOffsets names the alternate serialized order used by programs
// that store shift before scale in each modulation triplet.
type ShiftFirstGateOffsets struct {
	PreShift, PreScale, PreGate    int
	PostShift, PostScale, PostGate int
}

func PairedShiftFirstGateOffsets() ShiftFirstGateOffsets {
	return ShiftFirstGateOffsets{PreShift: 0, PreScale: 1, PreGate: 2, PostShift: 3, PostScale: 4, PostGate: 5}
}

// NormalizationProgram binds numerical guards shared by a compiled neural
// media pipeline. Values come from an artifact or repository recipe.
type NormalizationProgram struct {
	TransformerLayer float64 `json:"transformer_layer_epsilon"`
	FlowLayer        float64 `json:"flow_layer_epsilon"`
	TimeEmbeddingRMS float64 `json:"time_embedding_rms_epsilon"`
}

func (p NormalizationProgram) Validate() error {
	for _, value := range [...]float64{p.TransformerLayer, p.FlowLayer, p.TimeEmbeddingRMS} {
		if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return errors.New("media: invalid normalization program")
		}
	}
	return nil
}

// ValidateChannelMoments validates per-channel affine normalization vectors.
func ValidateChannelMoments[T ~float32 | ~float64](mean, standardDeviation []T, channels int) error {
	if channels <= 0 || len(mean) != channels || len(standardDeviation) != channels {
		return errors.New("media: channel moments do not match geometry")
	}
	for index := range mean {
		if math.IsNaN(float64(mean[index])) || math.IsInf(float64(mean[index]), 0) ||
			standardDeviation[index] <= 0 || math.IsNaN(float64(standardDeviation[index])) || math.IsInf(float64(standardDeviation[index]), 0) {
			return errors.New("media: channel moments are invalid")
		}
	}
	return nil
}

// NormalizePlanarChannelsInto applies (input-mean)/standardDeviation to
// channel-major planar storage.
func NormalizePlanarChannelsInto(output, input, mean, standardDeviation []float32, channels, positions int) error {
	if err := ValidateChannelMoments(mean, standardDeviation, channels); err != nil {
		return err
	}
	if channels <= 0 || positions <= 0 || len(output) != channels*positions || len(input) != len(output) {
		return errors.New("media: planar normalization storage differs")
	}
	for channel := range channels {
		for position := range positions {
			index := channel*positions + position
			output[index] = (input[index] - mean[channel]) / standardDeviation[channel]
		}
	}
	return nil
}
