// Timestep conditioning: sinusoidal embedding -> Linear/SiLU/Linear head
// embedding -> SiLU/Linear block projection. Host boundary math (ported
// adaptive CompileTimestepConditioningInto), f64 accumulation with one
// rounding per output — bit-parity with the g2 capture engine.
package latentvideo

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
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
	if batch == 0 || freqDim <= 0 || freqDim%2 != 0 || dim <= 0 || w.Period <= 0 {
		return nil, nil, fmt.Errorf("timestep conditioning: bad geometry batch=%d freq=%d dim=%d period=%d", batch, freqDim, dim, w.Period)
	}
	if len(w.Embed0W) != dim*freqDim || len(w.Embed0B) != dim ||
		len(w.Embed2W) != dim*dim || len(w.Embed2B) != dim ||
		len(w.ProjectW) != 6*dim*dim || len(w.ProjectB) != 6*dim {
		return nil, nil, fmt.Errorf("timestep conditioning: projection shapes are invalid")
	}
	frequencies := make([]float32, batch*freqDim)
	half := freqDim / 2
	logPeriod := math.Log(float64(w.Period))
	for row, value := range timesteps {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, nil, fmt.Errorf("timestep conditioning: timestep %d is non-finite", row)
		}
		for i := 0; i < half; i++ {
			angle := value * math.Exp(-logPeriod*float64(i)/float64(half))
			frequencies[row*freqDim+i] = float32(math.Cos(angle))
			frequencies[row*freqDim+half+i] = float32(math.Sin(angle))
		}
	}
	work := make([]float32, batch*dim)
	headE = make([]float32, batch*dim)
	blockE = make([]float32, batch*6*dim)
	hostmath.LinearF64(work, frequencies, w.Embed0W, w.Embed0B, batch, freqDim, dim)
	hostmath.SiLUInPlace(work)
	hostmath.LinearF64(headE, work, w.Embed2W, w.Embed2B, batch, dim, dim)
	copy(work, headE)
	hostmath.SiLUInPlace(work)
	hostmath.LinearF64(blockE, work, w.ProjectW, w.ProjectB, batch, dim, 6*dim)
	return headE, blockE, nil
}
