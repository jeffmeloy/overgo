package sampling

import (
	"errors"
	"fmt"
	"math"
)

// ValidDraftLimits validates a bounded speculative sampling request.
func ValidDraftLimits(maximum int, minimumProbability float64) bool {
	return maximum > 0 && minimumProbability >= 0 && minimumProbability <= 1 && !math.IsNaN(minimumProbability)
}

// GreedyLogit returns the maximum-logit token and its normalized probability.
func GreedyLogit(logits []float32) (int, float64, error) {
	if len(logits) == 0 {
		return 0, 0, errors.New("sampling: greedy logits are empty")
	}
	best := 0
	maximum := float64(logits[0])
	if math.IsNaN(maximum) {
		return 0, 0, errors.New("sampling: greedy logits contain NaN")
	}
	for index := 1; index < len(logits); index++ {
		value := float64(logits[index])
		if math.IsNaN(value) {
			return 0, 0, errors.New("sampling: greedy logits contain NaN")
		}
		if value > maximum {
			maximum, best = value, index
		}
	}
	if math.IsInf(maximum, -1) {
		return 0, 0, errors.New("sampling: greedy logits contain no finite candidate")
	}
	var denominator float64
	for _, raw := range logits {
		value := float64(raw)
		if math.IsInf(value, 1) {
			if math.IsInf(maximum, 1) && value == maximum {
				denominator++
			}
			continue
		}
		denominator += math.Exp(value - maximum)
	}
	if denominator == 0 || math.IsNaN(denominator) {
		return 0, 0, fmt.Errorf("sampling: greedy probability normalization failed")
	}
	return best, 1 / denominator, nil
}
