// Timestep conditioning: sinusoidal embedding -> Linear/SiLU/Linear head
// embedding -> SiLU/Linear block projection. Host boundary math (ported
// adaptive CompileTimestepConditioningInto), f64 accumulation with one
// rounding per output — bit-parity with the g2 capture engine.
package latentvideo

import (
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
)

// TimestepConditioningWeights: the four projection stages.
type TimestepConditioningWeights struct {
	Embed0W, Embed0B     []float32 // [dim, freqDim] / [dim]
	Embed2W, Embed2B     []float32 // [dim, dim] / [dim]
	ProjectW, ProjectB   []float32 // [6*dim, dim] / [6*dim]
	FreqDim, Dim, Period int
}

// CompileTimestepConditioning: headE [batch*dim] (the raw time embedding fed
// to the output head) and blockE [batch*6*dim] (the per-block modulation
// conditioning) for one timestep batch.
func CompileTimestepConditioning(timesteps []float64, w TimestepConditioningWeights) (headE, blockE []float32, err error) {
	batch, freqDim, dim := len(timesteps), w.FreqDim, w.Dim
	modulationFields := media.DefaultPairedShiftScaleGateFields()
	modulationWidth := media.PairedShiftScaleGateWidth(dim)
	if !checked.PositiveInts(batch, freqDim, dim, w.Period) || !checked.EvenInt(freqDim) {
		return nil, nil, fmt.Errorf("timestep conditioning: bad geometry batch=%d freq=%d dim=%d period=%d", batch, freqDim, dim, w.Period)
	}
	if err := checked.Length(w.Embed0W, dim, freqDim); err != nil {
		return nil, nil, fmt.Errorf("timestep conditioning: projection shapes are invalid: %w", err)
	}
	if err := checked.Length(w.Embed0B, dim); err != nil {
		return nil, nil, fmt.Errorf("timestep conditioning: projection shapes are invalid: %w", err)
	}
	if err := checked.Length(w.Embed2W, dim, dim); err != nil {
		return nil, nil, fmt.Errorf("timestep conditioning: projection shapes are invalid: %w", err)
	}
	if err := checked.Length(w.Embed2B, dim); err != nil {
		return nil, nil, fmt.Errorf("timestep conditioning: projection shapes are invalid: %w", err)
	}
	if err := checked.Length(w.ProjectW, modulationFields, dim, dim); err != nil {
		return nil, nil, fmt.Errorf("timestep conditioning: projection shapes are invalid: %w", err)
	}
	if err := checked.Length(w.ProjectB, modulationFields, dim); err != nil {
		return nil, nil, fmt.Errorf("timestep conditioning: projection shapes are invalid")
	}
	frequencies := make([]float32, batch*freqDim)
	encoding := media.SinusoidalProgram{Dimensions: freqDim, FrequencyBase: float64(w.Period), InputScale: float64(tensor.SingletonExtent)}
	for row, value := range timesteps {
		encoded, err := encoding.Encode32(value)
		if err != nil {
			return nil, nil, fmt.Errorf("timestep conditioning: timestep %d is non-finite", row)
		}
		copy(frequencies[row*freqDim:], encoded)
	}
	work := make([]float32, batch*dim)
	headE = make([]float32, batch*dim)
	blockE = make([]float32, batch*modulationWidth)
	hostmath.LinearF64(work, frequencies, w.Embed0W, w.Embed0B, batch, freqDim, dim)
	hostmath.SiLUInPlace(work)
	hostmath.LinearF64(headE, work, w.Embed2W, w.Embed2B, batch, dim, dim)
	copy(work, headE)
	hostmath.SiLUInPlace(work)
	hostmath.LinearF64(blockE, work, w.ProjectW, w.ProjectB, batch, dim, modulationWidth)
	return headE, blockE, nil
}
