package media

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

// SinusoidalProgram is a compiled cos-first/sin-second timestep embedding.
// FrequencyBase and InputScale are supplied by the owning artifact recipe.
type SinusoidalProgram struct {
	Dimensions    int
	FrequencyBase float64
	InputScale    float64
}

func (p SinusoidalProgram) Validate() error {
	if p.Dimensions <= 0 || p.Dimensions%tensor.PairedExtent != 0 ||
		p.FrequencyBase <= 0 || p.InputScale <= 0 ||
		math.IsNaN(p.FrequencyBase) || math.IsInf(p.FrequencyBase, 0) ||
		math.IsNaN(p.InputScale) || math.IsInf(p.InputScale, 0) {
		return errors.New("media: invalid sinusoidal timestep program")
	}
	return nil
}

func (p SinusoidalProgram) Encode64(value float64) ([]float64, error) {
	if err := p.Validate(); err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		if err != nil {
			return nil, err
		}
		return nil, errors.New("media: timestep value is not finite")
	}
	half := p.Dimensions / tensor.PairedExtent
	encoded := make([]float64, p.Dimensions)
	for index := range half {
		frequency := math.Exp(-math.Log(p.FrequencyBase) * float64(index) / float64(half))
		angle := value * p.InputScale * frequency
		encoded[index] = math.Cos(angle)
		encoded[half+index] = math.Sin(angle)
	}
	return encoded, nil
}

func (p SinusoidalProgram) Encode32(value float64) ([]float32, error) {
	encoded, err := p.Encode64(value)
	if err != nil {
		return nil, err
	}
	result := make([]float32, len(encoded))
	for index, item := range encoded {
		result[index] = float32(item)
	}
	return result, nil
}
