package sampling

import (
	"math"
	"math/rand"
)

// AddGumbelNoise applies the diffusion sampler's temperature transform in
// place. The floor prevents log(0) for adversarial random sources.
func AddGumbelNoise(logits []float32, temperature float32, rng *rand.Rand) {
	if temperature == 0 {
		return
	}
	for index, logit := range logits {
		uniform := max(rng.Float64(), 1e-20)
		noise := math.Pow(-math.Log(uniform), float64(temperature))
		exponential := float32(math.Exp(float64(logit)))
		logits[index] = float32(float64(exponential) / noise)
	}
}

// Entropy returns the stable entropy of a sampled probability set.
func Entropy(result SampleProbabilityResult) float64 {
	var entropy float64
	for _, candidate := range result.Top {
		entropy -= candidate.Probability * math.Log(candidate.Probability+1e-10)
	}
	return entropy
}

// ProbabilityMargin returns the gap between the two leading candidates, or
// the only candidate's probability when the set is singular.
func ProbabilityMargin(result SampleProbabilityResult) float64 {
	if len(result.Top) == 0 {
		return 0
	}
	if len(result.Top) == 1 {
		return result.Top[0].Probability
	}
	return result.Top[0].Probability - result.Top[1].Probability
}
