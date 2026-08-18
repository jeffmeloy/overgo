package composition

import (
	"fmt"
	"math"

	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
)

// InjectionConfig: the pre-written residual-injection protocol. The drafter's
// mean hidden state at its 0.75-depth layer projects through a learned linear
// connector into the scorer's residual stream at the scorer's 0.75-depth
// layer, magnitude-rescaled at Alpha. The connector trains by feature
// matching -- ridge regression from drafter vectors onto the scorer's
// measured context-gap vectors -- never by end-to-end task cross-entropy,
// the recorded failure mode of the retired adapter tier.
type InjectionConfig struct {
	ScorerDir  string
	DrafterDir string
	Prefix     int
	Gap        int
	Target     int
	Alpha      float32
	Ridge      float64
	Train      [][]int // training windows (scorer tokens), each >= Prefix+Gap+Target
	HeldOut    [][]int // evaluation windows, disjoint from training
}

// InjectionWindowOutcome: one held-out window's measured arms.
type InjectionWindowOutcome struct {
	Window     int
	BaselineCE float64
	InjectedCE float64
}

// InjectionResult: verdict plus the training diagnostics that make a refusal
// interpretable.
type InjectionResult struct {
	ScorerLayer  int
	DrafterLayer int
	TrainWindows int
	FitResidual  float64 // mean squared residual of the ridge fit
	Outcomes     []InjectionWindowOutcome
	Ship         bool
	Reason       string
}

// RunInjectionViability executes the residual-injection experiment. It
// returns an error only when the experiment could not run; a refusal is a
// successful experiment whose verdict is Ship=false with the measured reason.
func RunInjectionViability(config InjectionConfig) (InjectionResult, error) {
	if config.Prefix < 2 || config.Gap < 1 || config.Target < 2 ||
		len(config.Train) < 2 || len(config.HeldOut) == 0 {
		return InjectionResult{}, fmt.Errorf("composition: injection prefix, gap, target, train and held-out windows are required")
	}
	if config.Alpha <= 0 || config.Alpha >= 1 {
		return InjectionResult{}, fmt.Errorf("composition: injection alpha must lie in (0,1)")
	}
	if config.Ridge <= 0 {
		return InjectionResult{}, fmt.Errorf("composition: ridge regularization must be positive")
	}
	scorer, err := densecausal.Load(config.ScorerDir)
	if err != nil {
		return InjectionResult{}, fmt.Errorf("composition: load scorer: %w", err)
	}
	drafter, err := densecausal.Load(config.DrafterDir)
	if err != nil {
		return InjectionResult{}, fmt.Errorf("composition: load drafter: %w", err)
	}
	scorerTok, err := hfbpe.Load(config.ScorerDir)
	if err != nil {
		return InjectionResult{}, fmt.Errorf("composition: load scorer tokenizer: %w", err)
	}
	drafterTok, err := hfbpe.Load(config.DrafterDir)
	if err != nil {
		return InjectionResult{}, fmt.Errorf("composition: load drafter tokenizer: %w", err)
	}
	span := config.Prefix + config.Gap + config.Target
	result := InjectionResult{
		ScorerLayer:  (scorer.Dims.Layers * 3) / 4,
		DrafterLayer: (drafter.Dims.Layers * 3) / 4,
		TrainWindows: len(config.Train),
	}
	// Training set: drafter vector z_w (mean hidden at the drafter layer over
	// its own tokenization of the prefix text) against the scorer's measured
	// context gap g_w (mean hidden difference at the scorer layer between the
	// full-context and prefix-only forwards).
	drafterDim, scorerDim := drafter.Dims.Hidden, scorer.Dims.Hidden
	inputs := make([][]float64, 0, len(config.Train))
	targets := make([][]float64, 0, len(config.Train))
	for index, window := range config.Train {
		if len(window) < span {
			return InjectionResult{}, fmt.Errorf("composition: train window %d has %d tokens, need %d", index, len(window), span)
		}
		z, err := drafterVector(drafter, drafterTok, scorerTok, window[:config.Prefix], result.DrafterLayer)
		if err != nil {
			return InjectionResult{}, fmt.Errorf("composition: train window %d drafter: %w", index, err)
		}
		gap, err := contextGap(scorer, window[:span], config.Prefix, config.Gap, result.ScorerLayer)
		if err != nil {
			return InjectionResult{}, fmt.Errorf("composition: train window %d gap: %w", index, err)
		}
		inputs = append(inputs, z)
		targets = append(targets, gap)
	}
	projection, residual, err := ridgeFit(inputs, targets, drafterDim, scorerDim, config.Ridge)
	if err != nil {
		return InjectionResult{}, err
	}
	result.FitResidual = residual
	shipped := 0
	for index, window := range config.HeldOut {
		if len(window) < span {
			return InjectionResult{}, fmt.Errorf("composition: held-out window %d has %d tokens, need %d", index, len(window), span)
		}
		context := window[:config.Prefix]
		target := window[config.Prefix+config.Gap : span]
		evaluation := append(append([]int(nil), context...), target...)
		from := config.Prefix
		baseline, err := suffixCE(scorer, context, target)
		if err != nil {
			return InjectionResult{}, fmt.Errorf("composition: held-out %d baseline: %w", index, err)
		}
		z, err := drafterVector(drafter, drafterTok, scorerTok, context, result.DrafterLayer)
		if err != nil {
			return InjectionResult{}, fmt.Errorf("composition: held-out %d drafter: %w", index, err)
		}
		vector := make([]float32, scorerDim)
		for row := 0; row < scorerDim; row++ {
			sum := float64(0)
			for column := 0; column < drafterDim; column++ {
				sum += projection[row*drafterDim+column] * z[column]
			}
			vector[row] = float32(sum)
		}
		injected, err := scorer.LossRangeInjected(evaluation, from, result.ScorerLayer, vector, config.Alpha)
		if err != nil {
			return InjectionResult{}, fmt.Errorf("composition: held-out %d injected: %w", index, err)
		}
		result.Outcomes = append(result.Outcomes, InjectionWindowOutcome{
			Window: index, BaselineCE: baseline, InjectedCE: injected,
		})
		if injected < baseline {
			shipped++
		}
	}
	if shipped == len(config.HeldOut) {
		result.Ship = true
		result.Reason = fmt.Sprintf(
			"envelope separation: injected held-out CE beats the un-injected baseline on all %d windows (fit residual %.6f)",
			len(config.HeldOut), residual)
	} else {
		result.Reason = fmt.Sprintf(
			"refusal: injection beat baseline on %d of %d held-out windows (fit residual %.6f); failure mode: the ridge-fit connector did not transfer the context gap",
			shipped, len(config.HeldOut), residual)
	}
	return result, nil
}

// drafterVector: the drafter's mean hidden state at its injection-depth layer
// over its OWN tokenization of the prefix text -- knowledge crosses the
// tokenizer boundary as text, exactly like the Tier-0 chain.
func drafterVector(
	drafter *densecausal.Model,
	drafterTok, scorerTok *hfbpe.Tokenizer,
	prefix []int,
	layer int,
) ([]float64, error) {
	text := scorerTok.Decode(prefix)
	tokens, err := drafterTok.Encode(text)
	if err != nil {
		return nil, err
	}
	if len(tokens) < 2 {
		return nil, fmt.Errorf("composition: drafter prefix collapsed to %d tokens", len(tokens))
	}
	states, err := drafter.LayerStates(tokens)
	if err != nil {
		return nil, err
	}
	hidden := drafter.Dims.Hidden
	vector := make([]float64, hidden)
	layerState := states[layer]
	for position := 0; position < len(tokens); position++ {
		for index := 0; index < hidden; index++ {
			vector[index] += float64(layerState[position*hidden+index])
		}
	}
	for index := range vector {
		vector[index] /= float64(len(tokens))
	}
	return vector, nil
}

// contextGap: the scorer's mean hidden difference at the injection layer,
// measured AT THE TARGET POSITIONS between the with-gap forward (prefix ++
// gap ++ target) and the gap-elided forward (prefix ++ target). Causal
// attention keeps pre-gap positions bit-identical between the two, so the
// elided information manifests exactly where the targets attend -- that
// difference is what the connector must supply.
func contextGap(scorer *densecausal.Model, window []int, prefix, gapLength, layer int) ([]float64, error) {
	span := len(window)
	targetLength := span - prefix - gapLength
	if targetLength < 1 {
		return nil, fmt.Errorf("composition: window leaves no target positions")
	}
	withGapStates, err := scorer.LayerStates(window)
	if err != nil {
		return nil, err
	}
	elided := append(append([]int(nil), window[:prefix]...), window[prefix+gapLength:]...)
	elidedStates, err := scorer.LayerStates(elided)
	if err != nil {
		return nil, err
	}
	hidden := scorer.Dims.Hidden
	gap := make([]float64, hidden)
	full, short := withGapStates[layer], elidedStates[layer]
	for position := 0; position < targetLength; position++ {
		fullOffset := (prefix + gapLength + position) * hidden
		elidedOffset := (prefix + position) * hidden
		for index := 0; index < hidden; index++ {
			gap[index] += float64(full[fullOffset+index]) - float64(short[elidedOffset+index])
		}
	}
	for index := range gap {
		gap[index] /= float64(targetLength)
	}
	return gap, nil
}

// ridgeFit solves the per-output-dimension ridge regression P = argmin
// sum_w |P z_w - g_w|^2 + ridge |P|^2 via the (d x d) normal equations in
// FP64. Deterministic: Gaussian elimination with partial pivoting.
func ridgeFit(inputs, targets [][]float64, inputDim, outputDim int, ridge float64) ([]float64, float64, error) {
	samples := len(inputs)
	gram := make([]float64, inputDim*inputDim)
	for _, z := range inputs {
		for i := 0; i < inputDim; i++ {
			for j := i; j < inputDim; j++ {
				gram[i*inputDim+j] += z[i] * z[j]
			}
		}
	}
	for i := 0; i < inputDim; i++ {
		for j := 0; j < i; j++ {
			gram[i*inputDim+j] = gram[j*inputDim+i]
		}
		gram[i*inputDim+i] += ridge
	}
	factored := append([]float64(nil), gram...)
	pivots, err := luFactor(factored, inputDim)
	if err != nil {
		return nil, 0, err
	}
	projection := make([]float64, outputDim*inputDim)
	rhs := make([]float64, inputDim)
	for out := 0; out < outputDim; out++ {
		for i := range rhs {
			rhs[i] = 0
		}
		for sample, z := range inputs {
			for i := 0; i < inputDim; i++ {
				rhs[i] += z[i] * targets[sample][out]
			}
		}
		luSolve(factored, pivots, rhs, inputDim)
		copy(projection[out*inputDim:(out+1)*inputDim], rhs)
	}
	residual := float64(0)
	for sample, z := range inputs {
		for out := 0; out < outputDim; out++ {
			predicted := float64(0)
			for i := 0; i < inputDim; i++ {
				predicted += projection[out*inputDim+i] * z[i]
			}
			delta := predicted - targets[sample][out]
			residual += delta * delta
		}
	}
	residual /= float64(samples * outputDim)
	if math.IsNaN(residual) || math.IsInf(residual, 0) {
		return nil, 0, fmt.Errorf("composition: ridge fit diverged")
	}
	return projection, residual, nil
}

func luFactor(matrix []float64, n int) ([]int, error) {
	pivots := make([]int, n)
	for column := 0; column < n; column++ {
		best, magnitude := column, math.Abs(matrix[column*n+column])
		for row := column + 1; row < n; row++ {
			if value := math.Abs(matrix[row*n+column]); value > magnitude {
				best, magnitude = row, value
			}
		}
		if magnitude == 0 {
			return nil, fmt.Errorf("composition: singular ridge system at column %d", column)
		}
		pivots[column] = best
		if best != column {
			for k := 0; k < n; k++ {
				matrix[column*n+k], matrix[best*n+k] = matrix[best*n+k], matrix[column*n+k]
			}
		}
		inverse := 1 / matrix[column*n+column]
		for row := column + 1; row < n; row++ {
			factor := matrix[row*n+column] * inverse
			matrix[row*n+column] = factor
			for k := column + 1; k < n; k++ {
				matrix[row*n+k] -= factor * matrix[column*n+k]
			}
		}
	}
	return pivots, nil
}

func luSolve(factored []float64, pivots []int, rhs []float64, n int) {
	for column := 0; column < n; column++ {
		if pivots[column] != column {
			rhs[column], rhs[pivots[column]] = rhs[pivots[column]], rhs[column]
		}
		for row := column + 1; row < n; row++ {
			rhs[row] -= factored[row*n+column] * rhs[column]
		}
	}
	for row := n - 1; row >= 0; row-- {
		for column := row + 1; column < n; column++ {
			rhs[row] -= factored[row*n+column] * rhs[column]
		}
		rhs[row] /= factored[row*n+row]
	}
}
