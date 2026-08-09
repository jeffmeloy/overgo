// Forward components for the patched time-series capability: input padding,
// incremental patch statistics, RevIN patch embedding through the tokenizer
// residual block, and the quantile output head. The decoder stack is the
// remaining forward slice; until it lands Forecast refuses loudly rather
// than producing unverified numbers.
//
// Semantics are ported behavior, written fresh: masks use 1 for padding and
// 0 for legitimate values; patch statistics are Welford parallel merges so
// patch i carries the running stats of patches 0..i; RevIN divides by the
// running sigma unless it is below the recorded tolerance, in which case the
// divisor is one; the head denormalizes with the LAST patch's statistics.
package seriesforecast

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// revinTolerance mirrors the adaptive execution-profile fact revin.tolerance
// for this artifact family. A sigma below this is treated as degenerate and
// the divisor becomes one. Ledger row candidate: derive from the artifact or
// profile data when the recipe carries it; recorded here as the port's one
// carried execution fact.
const revinTolerance = 1e-06

// padToPatches front-pads the series to a whole number of patches; returned
// masks mark padding with one. A nil mask input means all values legitimate.
func padToPatches(series, masks []float32, patchLen int) ([]float32, []float32, error) {
	if len(series) == 0 || patchLen <= 0 || (masks != nil && len(masks) != len(series)) {
		return nil, nil, fmt.Errorf("seriesforecast: input series=%d masks=%d patch=%d", len(series), len(masks), patchLen)
	}
	front := (patchLen - len(series)%patchLen) % patchLen
	elements := front + len(series)
	paddedMasks := make([]float32, elements)
	for i := 0; i < front; i++ {
		paddedMasks[i] = 1
	}
	copy(paddedMasks[front:], masks)
	if front == 0 {
		return series, paddedMasks, nil
	}
	padded := make([]float32, elements)
	copy(padded[front:], series)
	return padded, paddedMasks, nil
}

// patchStats fills mu/sigma per patch with the RUNNING statistics over all
// legitimate values seen through that patch (Welford parallel merge).
func patchStats(series, masks []float32, patchLen int, mu, sigma []float64) {
	var n, runningMu, runningSigma float64
	for i := range mu {
		var incN, incSum float64
		for j := 0; j < patchLen; j++ {
			if masks[i*patchLen+j] == 0 {
				incN++
				incSum += float64(series[i*patchLen+j])
			}
		}
		var incMu float64
		if incN > 0 {
			incMu = incSum / incN
		}
		var incVarNum float64
		for j := 0; j < patchLen; j++ {
			if masks[i*patchLen+j] == 0 {
				d := float64(series[i*patchLen+j]) - incMu
				incVarNum += d * d
			}
		}
		var incVar float64
		if incN > 0 {
			incVar = incVarNum / incN
		}
		incSigma := math.Sqrt(incVar)
		newN := n + incN
		var newMu, newSigma float64
		if newN > 0 {
			newMu = (n*runningMu + incMu*incN) / newN
			t1 := n * runningSigma * runningSigma
			t2 := incN * incSigma * incSigma
			t3 := n * (runningMu - newMu) * (runningMu - newMu)
			t4 := incN * (incMu - newMu) * (incMu - newMu)
			newVar := (t1 + t2 + t3 + t4) / newN
			if newVar < 0 {
				newVar = 0
			}
			newSigma = math.Sqrt(newVar)
		}
		n, runningMu, runningSigma = newN, newMu, newSigma
		mu[i], sigma[i] = runningMu, runningSigma
	}
}

// patchEmbed normalizes each patch (RevIN with the running stats), zeroes
// masked entries, concatenates [normed, mask], and runs the tokenizer
// residual block into out[token*hidden:].
func (m *Model) patchEmbed(out, series, masks []float32, mu, sigma []float64) error {
	p := m.Dims.PatchLen
	input := make([]float32, patchInputStreams*p)
	for i := range mu {
		denom := sigma[i]
		if denom < revinTolerance {
			denom = 1
		}
		for j := 0; j < p; j++ {
			normed := (float64(series[i*p+j]) - mu[i]) / denom
			if masks[i*p+j] != 0 {
				normed = 0
			}
			input[j] = float32(normed)
			input[p+j] = masks[i*p+j]
		}
		if err := m.residualBlock(out[i*m.Dims.Hidden:(i+1)*m.Dims.Hidden], "tokenizer", input); err != nil {
			return err
		}
	}
	return nil
}

// residualBlock: dst = W_out·SiLU(W_hidden·x + b_hidden) + b_out
// + (W_residual·x + b_residual). Biases are optional (the output heads ship
// without them).
func (m *Model) residualBlock(dst []float32, prefix string, x []float32) error {
	hiddenShape, ok := m.Shapes[prefix+".hidden_layer.weight"]
	if !ok || len(hiddenShape) != 2 {
		return fmt.Errorf("seriesforecast: residual block %q hidden layer missing", prefix)
	}
	outputShape, ok := m.Shapes[prefix+".output_layer.weight"]
	if !ok || len(outputShape) != 2 {
		return fmt.Errorf("seriesforecast: residual block %q output layer missing", prefix)
	}
	hiddenDim, inputDim, outputDim := hiddenShape[0], hiddenShape[1], outputShape[0]
	if len(x) != inputDim || len(dst) != outputDim {
		return fmt.Errorf("seriesforecast: residual block %q x=%d dst=%d want in=%d out=%d", prefix, len(x), len(dst), inputDim, outputDim)
	}
	hidden := make([]float32, hiddenDim)
	hostmath.Linear(hidden, x, m.Weights[prefix+".hidden_layer.weight"], 1, inputDim, hiddenDim)
	hostmath.AddBias(hidden, m.Weights[prefix+".hidden_layer.bias"])
	hostmath.SiLUInPlace(hidden)
	hostmath.Linear(dst, hidden, m.Weights[prefix+".output_layer.weight"], 1, hiddenDim, outputDim)
	hostmath.AddBias(dst, m.Weights[prefix+".output_layer.bias"])
	residual := make([]float32, outputDim)
	hostmath.Linear(residual, x, m.Weights[prefix+".residual_layer.weight"], 1, inputDim, outputDim)
	hostmath.AddBias(residual, m.Weights[prefix+".residual_layer.bias"])
	for k := range dst {
		dst[k] += residual[k]
	}
	return nil
}

// Forecast runs the full forward: pad, running patch stats, RevIN embed,
// the post-norm decoder stack, then the quantile head denormalized with the
// last patch's statistics. The result is [Horizon*Quantiles] flat, t-major.
func (m *Model) Forecast(series []float32) ([]float32, error) {
	padded, masks, err := padToPatches(series, make([]float32, len(series)), m.Dims.PatchLen)
	if err != nil {
		return nil, err
	}
	tokens := len(padded) / m.Dims.PatchLen
	mu := make([]float64, tokens)
	sigma := make([]float64, tokens)
	patchStats(padded, masks, m.Dims.PatchLen, mu, sigma)
	hidden := make([]float32, tokens*m.Dims.Hidden)
	if err := m.patchEmbed(hidden, padded, masks, mu, sigma); err != nil {
		return nil, err
	}
	invFreq := hostmath.RopeInvFreq(m.Dims.RopeTheta, m.Dims.HeadDim)
	for index := 0; index < m.Dims.Layers; index++ {
		l, err := m.layerWeights(index)
		if err != nil {
			return nil, err
		}
		m.layerForward(hidden, l, invFreq, tokens)
	}
	out := make([]float32, m.Dims.Horizon*m.Dims.Quantiles)
	last := tokens - 1
	if err := m.head(out, hidden[last*m.Dims.Hidden:(last+1)*m.Dims.Hidden], mu[last], sigma[last]); err != nil {
		return nil, err
	}
	return out, nil
}

// head runs the point-projection residual block on the final token and
// denormalizes with the last patch's running statistics.
func (m *Model) head(dst, lastToken []float32, mu, sigma float64) error {
	if err := m.residualBlock(dst, "output_projection_point", lastToken); err != nil {
		return err
	}
	for k := range dst {
		dst[k] = float32(float64(dst[k])*sigma + mu)
	}
	return nil
}
