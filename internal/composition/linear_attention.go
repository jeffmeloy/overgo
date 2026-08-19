// Linear-attention connector: content-addressed retrieval over drafter
// states, the one mechanism the refuted mean-vector ridge lacked.
package composition

import (
	"fmt"
	"math"

	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
)

// RunLinearAttentionViability retries the injection protocol with exactly
// one mechanism changed: instead of injecting the ridge projection of the
// drafter's MEAN hidden state at every target position, each target
// retrieves its own vector by linear attention over the drafter's per-token
// states. Keys and values are the drafter states projected through the same
// ridge fit the mean protocol trains (no new fitted parameters); queries are
// the scorer's own un-injected hidden states at the target positions; the
// kernel is the standard elu+1 feature map. If content addressing is what
// the linear connector lacked, this ships; if it refuses beside the mean
// protocol, the linear connector family is refuted at this budget with or
// without attention.
func RunLinearAttentionViability(config InjectionConfig) (InjectionResult, error) {
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
		memory, err := drafterProjectedStates(drafter, drafterTok, scorerTok, context, result.DrafterLayer, projection, drafterDim, scorerDim)
		if err != nil {
			return InjectionResult{}, fmt.Errorf("composition: held-out %d drafter states: %w", index, err)
		}
		queries, err := scorerTargetStates(scorer, evaluation, from, result.ScorerLayer)
		if err != nil {
			return InjectionResult{}, fmt.Errorf("composition: held-out %d queries: %w", index, err)
		}
		vectors := make([][]float32, len(queries))
		for position, query := range queries {
			vectors[position] = linearAttentionReadout(query, memory, scorerDim)
		}
		injected, err := scorer.LossRangeInjectedVectors(evaluation, from, result.ScorerLayer, vectors, config.Alpha)
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
			"envelope separation: content-addressed injection beats the un-injected baseline on all %d windows (fit residual %.6f)",
			len(config.HeldOut), residual)
	} else {
		result.Reason = fmt.Sprintf(
			"refusal: linear-attention injection beat baseline on %d of %d held-out windows (fit residual %.6f); failure mode: content addressing did not rescue the linear connector",
			shipped, len(config.HeldOut), residual)
	}
	return result, nil
}

// drafterProjectedStates: the drafter's per-token hidden states over its own
// tokenization of the prefix text, each projected into scorer space through
// the ridge fit. These are the connector's keys and values.
func drafterProjectedStates(
	drafter *densecausal.Model,
	drafterTok, scorerTok *hfbpe.Tokenizer,
	prefix []int,
	layer int,
	projection []float64,
	drafterDim, scorerDim int,
) ([][]float64, error) {
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
	layerState := states[layer]
	projected := make([][]float64, len(tokens))
	for position := range tokens {
		vector := make([]float64, scorerDim)
		for row := 0; row < scorerDim; row++ {
			sum := float64(0)
			for column := 0; column < drafterDim; column++ {
				sum += projection[row*drafterDim+column] * float64(layerState[position*drafterDim+column])
			}
			vector[row] = sum
		}
		projected[position] = vector
	}
	return projected, nil
}

// scorerTargetStates: the scorer's own un-injected hidden states at the
// scored positions of the evaluation sequence -- the connector's queries.
func scorerTargetStates(scorer *densecausal.Model, evaluation []int, from, layer int) ([][]float64, error) {
	states, err := scorer.LayerStates(evaluation)
	if err != nil {
		return nil, err
	}
	hidden := scorer.Dims.Hidden
	layerState := states[layer]
	queries := make([][]float64, len(evaluation)-from)
	for position := from; position < len(evaluation); position++ {
		query := make([]float64, hidden)
		for index := 0; index < hidden; index++ {
			query[index] = float64(layerState[position*hidden+index])
		}
		queries[position-from] = query
	}
	return queries, nil
}

// linearAttentionReadout: the elu+1 kernel readout over the projected
// drafter memory -- out = sum_i phi(q)*phi(k_i) v_i / sum_i phi(q)*phi(k_i)
// with keys equal to values, all in scorer space. Deterministic FP64.
func linearAttentionReadout(query []float64, memory [][]float64, dim int) []float32 {
	phi := func(value float64) float64 {
		if value > 0 {
			return value + 1
		}
		return math.Exp(value)
	}
	readout := make([]float64, dim)
	normalizer := float64(0)
	for _, key := range memory {
		score := float64(0)
		for index := 0; index < dim; index++ {
			score += phi(query[index]) * phi(key[index])
		}
		normalizer += score
		for index := 0; index < dim; index++ {
			readout[index] += score * key[index]
		}
	}
	vector := make([]float32, dim)
	if normalizer <= 0 || math.IsNaN(normalizer) || math.IsInf(normalizer, 0) {
		return vector
	}
	for index := 0; index < dim; index++ {
		vector[index] = float32(readout[index] / normalizer)
	}
	return vector
}
