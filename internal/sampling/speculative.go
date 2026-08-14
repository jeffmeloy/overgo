package sampling

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// SpeculativeSampleResult: committed token and verification probabilities.
type SpeculativeSampleResult struct {
	Token             int
	Accepted          bool
	TargetProbability float64
	DraftProbability  float64
	Target            []TokenProbability
}

// SpeculativeSample: ratio acceptance plus positive-residual correction.
func (s *Sampler) SpeculativeSample(
	logits []float32,
	history []int,
	draft []TokenProbability,
	proposed int,
) (SpeculativeSampleResult, error) {
	if s == nil {
		return SpeculativeSampleResult{}, errors.New("sampler is nil")
	}
	if len(logits) == 0 {
		return SpeculativeSampleResult{}, errors.New("sampling logits are empty")
	}
	draftDense, draftProbability, err := validateDraftDistribution(
		draft, proposed, len(logits),
	)
	if err != nil {
		return SpeculativeSampleResult{}, err
	}
	candidates, total, original, mirostat, err := s.speculativeCandidates(logits, history)
	if err != nil {
		return SpeculativeSampleResult{}, err
	}
	target := make([]float64, len(logits))
	for _, item := range candidates {
		target[item.id] = item.probability / total
	}
	targetProbability := target[proposed]
	accepted := targetProbability >= draftProbability
	if len(candidates) > 1 {
		accepted = s.random.Float64() < math.Min(1, targetProbability/draftProbability)
	}
	chosen := proposed
	if !accepted {
		residual := make([]candidate, 0, len(candidates))
		var residualTotal float64
		for _, item := range candidates {
			probability := math.Max(0, target[item.id]-draftDense[item.id])
			if probability == 0 {
				continue
			}
			residual = append(residual, candidate{id: item.id, probability: probability})
			residualTotal += probability
		}
		if residualTotal <= 0 || math.IsNaN(residualTotal) || math.IsInf(residualTotal, 0) {
			return SpeculativeSampleResult{}, errors.New(
				"sampling speculative residual distribution is empty",
			)
		}
		chosen = residual[sampleCandidateIndex(
			s.random.Float64(), residual, residualTotal,
		)].id
	}
	if err := s.commitSpeculativeToken(chosen, target[chosen], original, mirostat); err != nil {
		return SpeculativeSampleResult{}, err
	}
	return SpeculativeSampleResult{
		Token: chosen, Accepted: accepted,
		TargetProbability: targetProbability, DraftProbability: draftProbability,
		Target: sortedTokenProbabilities(candidates, total),
	}, nil
}

func (s *Sampler) speculativeCandidates(
	logits []float32,
	history []int,
) ([]candidate, float64, []float64, bool, error) {
	adjusted := s.copyLogits(logits)
	for index, value := range adjusted {
		if math.IsNaN(float64(value)) {
			return nil, 0, nil, false, fmt.Errorf("sampling logit %d is NaN", index)
		}
	}
	for index, bias := range s.config.LogitBiases {
		if bias.Token >= len(adjusted) {
			return nil, 0, nil, false, fmt.Errorf(
				"sampling logit bias %d token %d exceeds vocabulary size %d",
				index, bias.Token, len(adjusted),
			)
		}
		if math.IsInf(float64(bias.Bias), -1) {
			adjusted[bias.Token] = float32(math.Inf(-1))
		} else {
			adjusted[bias.Token] += bias.Bias
		}
	}
	if err := s.applyGrammar(adjusted); err != nil {
		return nil, 0, nil, false, err
	}
	if s.config.Mirostat != 0 {
		candidates, total, err := s.speculativeMirostatCandidates(adjusted)
		return candidates, total, nil, s.config.Temperature != 0, err
	}
	candidates := s.candidates(len(adjusted))
	for index, value := range adjusted {
		candidates[index] = candidate{id: index, scaledLogit: float64(value)}
	}
	useAdaptive := false
	for _, stage := range s.config.Samplers {
		if stage == SamplerAdaptiveP {
			useAdaptive = true
			continue
		}
		var err error
		candidates, err = s.applySamplerStage(candidates, len(logits), history, stage)
		if err != nil {
			return nil, 0, nil, false, err
		}
		if len(candidates) == 0 {
			return nil, 0, nil, false, fmt.Errorf(
				"sampling sampler %q removed every candidate", stage,
			)
		}
	}
	if useAdaptive {
		original, total, err := s.adaptiveCandidates(candidates, len(logits))
		if err != nil {
			return nil, 0, nil, false, err
		}
		return candidates, total, original, false, nil
	}
	total, err := candidateProbabilities(candidates)
	return candidates, total, nil, false, err
}

func (s *Sampler) speculativeMirostatCandidates(logits []float32) ([]candidate, float64, error) {
	if s.config.Temperature == 0 {
		best := 0
		for index := 1; index < len(logits); index++ {
			if logits[index] > logits[best] {
				best = index
			}
		}
		return []candidate{{id: best, probability: 1}}, 1, nil
	}
	candidates, total, err := normalizedCandidatesInto(
		s.candidates(len(logits)), logits, s.config.Temperature,
	)
	if err != nil {
		return nil, 0, err
	}
	keep := 1
	if s.config.Mirostat == 1 {
		const estimateTokens = 100
		estimateCount := min(estimateTokens-1, len(candidates)-1)
		var sumTIBI, sumTISquared float64
		for index := range estimateCount {
			ti := math.Log(float64(index+2) / float64(index+1))
			left := candidates[index].probability / total
			right := candidates[index+1].probability / total
			if left <= 0 || right <= 0 {
				continue
			}
			bi := math.Log(left / right)
			sumTIBI += ti * bi
			sumTISquared += ti * ti
		}
		if sumTISquared > 0 {
			sHat := sumTIBI / sumTISquared
			epsilonHat := sHat - 1
			denominator := 1 - math.Pow(float64(len(logits)), -epsilonHat)
			k := math.Pow(epsilonHat*math.Exp2(s.mu)/denominator, 1/sHat)
			if !math.IsNaN(k) && !math.IsInf(k, 0) && k > 1 {
				if k >= float64(len(candidates)) {
					keep = len(candidates)
				} else {
					keep = int(k)
				}
			}
		}
	} else {
		keep = 0
		for index := range candidates {
			probability := candidates[index].probability / total
			if -math.Log2(probability) > s.mu {
				break
			}
			keep = index + 1
		}
		if keep == 0 {
			keep = 1
		}
	}
	candidates = candidates[:keep]
	total = 0
	for _, item := range candidates {
		total += item.probability
	}
	return candidates, total, nil
}

func (s *Sampler) commitSpeculativeToken(
	token int,
	probability float64,
	original []float64,
	mirostat bool,
) error {
	if probability <= 0 || math.IsNaN(probability) || math.IsInf(probability, 0) {
		return errors.New("sampling speculative token has invalid target probability")
	}
	if _, err := s.acceptGrammar(token); err != nil {
		return err
	}
	if mirostat {
		observedSurprise := -math.Log2(probability)
		s.mu -= float64(s.config.MirostatEta) *
			(observedSurprise - float64(s.config.MirostatTau))
	}
	if original != nil && s.config.AdaptiveTarget >= 0 {
		s.adaptiveSum = original[token] + float64(s.config.AdaptiveDecay)*s.adaptiveSum
		s.adaptiveWeight = 1 + float64(s.config.AdaptiveDecay)*s.adaptiveWeight
	}
	return nil
}

func validateDraftDistribution(
	draft []TokenProbability,
	proposed int,
	vocabularySize int,
) ([]float64, float64, error) {
	if proposed < 0 || proposed >= vocabularySize || len(draft) == 0 {
		return nil, 0, errors.New("sampling speculative draft is invalid")
	}
	dense := make([]float64, vocabularySize)
	var total float64
	for index, item := range draft {
		if item.ID < 0 || item.ID >= vocabularySize || dense[item.ID] != 0 ||
			item.Probability <= 0 || math.IsNaN(item.Probability) ||
			math.IsInf(item.Probability, 0) {
			return nil, 0, fmt.Errorf("sampling speculative draft probability %d is invalid", index)
		}
		dense[item.ID] = item.Probability
		total += item.Probability
	}
	if math.Abs(total-1) > 1e-6 || dense[proposed] == 0 {
		return nil, 0, errors.New("sampling speculative draft distribution is not normalized")
	}
	return dense, dense[proposed], nil
}

func sortedTokenProbabilities(candidates []candidate, total float64) []TokenProbability {
	result := make([]TokenProbability, 0, len(candidates))
	for _, item := range candidates {
		probability := item.probability / total
		if probability > 0 {
			result = append(result, TokenProbability{ID: item.id, Probability: probability})
		}
	}
	slices.SortStableFunc(result, func(left, right TokenProbability) int {
		if left.Probability == right.Probability {
			return left.ID - right.ID
		}
		if left.Probability > right.Probability {
			return -1
		}
		return 1
	})
	return result
}
