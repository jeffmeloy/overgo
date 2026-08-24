package sampling

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"

	"overgo/internal/checked"
)

const (
	maxAdaptiveDecay    = float32(0.99)
	xtcDisableThreshold = float32(0.5)
	adaptiveTailCutoff  = 0.2
	dynamicRangeWidth   = 0.3
	adaptiveSharpness   = 10.0
	adaptivePeakLogit   = adaptiveSharpness / 2
	dryBaseTolerance    = 1.000001
)

const (
	mirostatDisabled = iota
	mirostatV1
	mirostatV2
)

type LogitBias struct {
	Token int
	Bias  float32
}

// BannedLogit returns the exact excluded-token value.
func BannedLogit() float32 { return float32(math.Inf(-1)) }

type InfillVocabulary struct {
	Pieces    []string
	EOG       []bool
	EOT       int
	EOS       int
	signature uint64
}

type Config struct {
	Temperature       float32
	DynatempRange     float32
	DynatempExponent  float32
	TopK              int
	TopP              float32
	MinP              float32
	TypicalP          float32
	TopNSigma         float32
	XTCProbability    float32
	XTCThreshold      float32
	MinKeep           int
	AdaptiveTarget    float32
	AdaptiveDecay     float32
	RepeatLastN       int
	RepeatPenalty     float32
	PresencePenalty   float32
	FrequencyPenalty  float32
	NoRepeatNgramSize int
	NgramWindow       int
	DryMultiplier     float32
	DryBase           float32
	DryAllowedLength  int
	DryPenaltyLastN   int
	DryBreakers       [][]int
	Mirostat          int
	MirostatTau       float32
	MirostatEta       float32
	Seed              int64
	Grammar           *TokenGrammar
	GBNF              *GBNFGrammar
	Samplers          []SamplerStage
	LogitBiases       []LogitBias
	Infill            *InfillVocabulary
}

type Sampler struct {
	config           Config
	random           *rand.Rand
	source           *splitMixSource
	mu               float64
	dryBreakers      map[int][][]int
	grammar          *TokenGrammar
	grammarState     int
	gbnf             *GBNFGrammar
	gbnfState        gbnfState
	gbnfHistory      []int
	adaptive         bool
	adaptiveSum      float64
	adaptiveWeight   float64
	probabilityLimit int
	lastProbability  SampleProbabilityResult
	adjustedScratch  []float32
	logitScratch     []float32
	candidateScratch []candidate
	topKSeen         map[int]struct{}
	adaptiveScratch  []float64
	penaltyCounts    map[int]int
	dryReversed      []int
	dryRepeatCount   []int
	dryMaxRepeat     map[int]int
	infillScratch    [2][]candidate
}

type candidate struct {
	id          int
	scaledLogit float64
	probability float64
}

type TokenProbability struct {
	ID          int
	Probability float64
}

type SampleProbabilityResult struct {
	Token               int
	SelectedProbability float64
	Top                 []TokenProbability
}

func New(config Config) (*Sampler, error) {
	if config.Grammar != nil && config.GBNF != nil {
		return nil, errors.New("sampling token grammar and GBNF are mutually exclusive")
	}
	if err := validateGrammar(config.Grammar); err != nil {
		return nil, fmt.Errorf("sampling grammar: %w", err)
	}
	if err := validateGBNFGrammar(config.GBNF); err != nil {
		return nil, fmt.Errorf("sampling GBNF: %w", err)
	}
	config.Grammar = cloneGrammar(config.Grammar)
	config.Samplers = cloneSamplerOrder(config.Samplers)
	config.LogitBiases = slices.Clone(config.LogitBiases)
	config.Infill = cloneInfillVocabulary(config.Infill)
	for index, stage := range config.Samplers {
		if err := validateSamplerStage(stage); err != nil {
			return nil, fmt.Errorf("sampling sampler stage %d: %w", index, err)
		}
	}
	if slices.Contains(config.Samplers, SamplerInfill) {
		if err := validateInfillVocabulary(config.Infill); err != nil {
			return nil, fmt.Errorf("sampling infill: %w", err)
		}
		config.Infill.signature = infillVocabularySignature(config.Infill)
	}
	for index, bias := range config.LogitBiases {
		if bias.Token < 0 {
			return nil, fmt.Errorf("sampling logit bias %d has negative token", index)
		}
		if bias.Bias != BannedLogit() && !checked.Finite32(bias.Bias) {
			return nil, fmt.Errorf("sampling logit bias %d must be finite or negative infinity", index)
		}
	}
	if !checked.NonNegativeFinite32(config.Temperature) {
		return nil, errors.New("sampling temperature must be non-negative")
	}
	if !checked.NonNegativeFinite32(config.DynatempRange) {
		return nil, errors.New("sampling dynamic-temperature range must be finite and non-negative")
	}
	if !checked.Finite32(config.DynatempExponent) {
		return nil, errors.New("sampling dynamic-temperature exponent must be finite")
	}
	if config.TopK < 0 {
		return nil, errors.New("sampling top-k must be non-negative")
	}
	if config.TopP < 0 || config.TopP > 1 || !checked.Finite32(config.TopP) {
		return nil, errors.New("sampling top-p must be in [0,1]")
	}
	if config.MinP < 0 || config.MinP > 1 || !checked.Finite32(config.MinP) {
		return nil, errors.New("sampling min-p must be in [0,1]")
	}
	if config.TypicalP < 0 || config.TypicalP > 1 || !checked.Finite32(config.TypicalP) {
		return nil, errors.New("sampling typical-p must be in [0,1]")
	}
	if !checked.Finite32(config.TopNSigma) {
		return nil, errors.New("sampling top-n-sigma must be finite")
	}
	if config.XTCProbability < 0 || config.XTCProbability > 1 ||
		!checked.Finite32(config.XTCProbability) {
		return nil, errors.New("sampling XTC probability must be in [0,1]")
	}
	if config.XTCThreshold < 0 || config.XTCThreshold > 1 ||
		!checked.Finite32(config.XTCThreshold) {
		return nil, errors.New("sampling XTC threshold must be in [0,1]")
	}
	if config.MinKeep < 0 {
		return nil, errors.New("sampling min-keep must be non-negative")
	}
	if !checked.Finite32(config.AdaptiveTarget) || config.AdaptiveTarget > 1 {
		return nil, errors.New("sampling adaptive-p target must be finite and at most 1")
	}
	if config.AdaptiveDecay < 0 || config.AdaptiveDecay > maxAdaptiveDecay ||
		!checked.Finite32(config.AdaptiveDecay) {
		return nil, errors.New("sampling adaptive-p decay must be in [0,0.99]")
	}
	if config.RepeatLastN < -1 {
		return nil, errors.New("sampling repeat-last-n must be at least -1")
	}
	if config.RepeatPenalty < 0 || !checked.Finite32(config.RepeatPenalty) ||
		((config.RepeatLastN != 0 || config.PresencePenalty != 0 || config.FrequencyPenalty != 0) &&
			config.RepeatPenalty == 0) {
		return nil, errors.New("sampling repeat penalty must be positive")
	}
	if !checked.Finite32(config.PresencePenalty) {
		return nil, errors.New("sampling presence penalty must be finite")
	}
	if !checked.Finite32(config.FrequencyPenalty) {
		return nil, errors.New("sampling frequency penalty must be finite")
	}
	if config.NoRepeatNgramSize < 0 || config.NgramWindow < 0 {
		return nil, errors.New("sampling no-repeat n-gram size and window must be non-negative")
	}
	if !checked.NonNegativeFinite32(config.DryMultiplier) {
		return nil, errors.New("sampling DRY multiplier must be finite and non-negative")
	}
	if config.DryMultiplier > 0 && (config.DryBase < 1 || !checked.Finite32(config.DryBase)) {
		return nil, errors.New("sampling DRY base must be finite and at least 1")
	}
	if config.DryAllowedLength < 0 {
		return nil, errors.New("sampling DRY allowed length must be non-negative")
	}
	if config.DryPenaltyLastN < -1 {
		return nil, errors.New("sampling DRY penalty-last-n must be at least -1")
	}
	dryBreakers := make(map[int][][]int, len(config.DryBreakers))
	config.DryBreakers = cloneBreakers(config.DryBreakers)
	for index, breaker := range config.DryBreakers {
		if len(breaker) == 0 {
			return nil, fmt.Errorf("sampling DRY breaker %d is empty", index)
		}
		for _, token := range breaker {
			if token < 0 {
				return nil, fmt.Errorf("sampling DRY breaker %d contains a negative token", index)
			}
		}
		head := breaker[0]
		dryBreakers[head] = append(dryBreakers[head], breaker[1:])
	}
	if config.Mirostat < mirostatDisabled || config.Mirostat > mirostatV2 {
		return nil, errors.New("sampling Mirostat version must be 0, 1, or 2")
	}
	if config.Mirostat != mirostatDisabled && !checked.PositiveFinite32(config.MirostatTau) {
		return nil, errors.New("sampling Mirostat tau must be finite and positive")
	}
	if config.Mirostat != mirostatDisabled && !checked.PositiveFinite32(config.MirostatEta) {
		return nil, errors.New("sampling Mirostat eta must be finite and positive")
	}
	source := &splitMixSource{}
	source.Seed(config.Seed)
	sampler := &Sampler{
		config:      config,
		source:      source,
		dryBreakers: dryBreakers,
		grammar:     cloneGrammar(config.Grammar),
		gbnf:        config.GBNF,
	}
	if sampler.grammar != nil {
		sampler.grammarState = sampler.grammar.Start
	}
	if sampler.gbnf != nil {
		state, stateErr := sampler.gbnf.initialState()
		if stateErr != nil {
			return nil, fmt.Errorf("sampling GBNF initial state: %w", stateErr)
		}
		sampler.gbnfState = state
	}
	sampler.random = rand.New(source)
	sampler.mu = 2 * float64(config.MirostatTau)
	for _, stage := range config.Samplers {
		if stage == SamplerAdaptiveP {
			sampler.adaptive = true
			break
		}
	}
	sampler.resetAdaptive()
	return sampler, nil
}

func (s *Sampler) Config() Config {
	if s == nil {
		return Config{}
	}
	result := s.config
	result.DryBreakers = cloneBreakers(result.DryBreakers)
	result.Grammar = cloneGrammar(result.Grammar)
	result.Samplers = cloneSamplerOrder(result.Samplers)
	result.LogitBiases = slices.Clone(result.LogitBiases)
	result.Infill = cloneInfillVocabulary(result.Infill)
	return result
}

// IsRawGreedy: exact device-argmax compatibility.
func (s *Sampler) IsRawGreedy() bool {
	if s == nil {
		return false
	}
	config := s.config
	return config.Temperature == 0 && config.DynatempRange == 0 && config.Mirostat == mirostatDisabled &&
		(config.RepeatPenalty == 0 || config.RepeatPenalty == 1) && config.PresencePenalty == 0 &&
		config.FrequencyPenalty == 0 && config.NoRepeatNgramSize == 0 && config.DryMultiplier == 0 &&
		config.XTCProbability == 0 && config.Grammar == nil && config.GBNF == nil &&
		len(config.LogitBiases) == 0 && config.Infill == nil &&
		(len(config.Samplers) == 0 || slices.Contains(config.Samplers, SamplerTemperature)) &&
		!slices.Contains(config.Samplers, SamplerAdaptiveP) &&
		!slices.Contains(config.Samplers, SamplerInfill)
}

// BoundedTopK: exact prefix-filter candidate limit.
func (s *Sampler) BoundedTopK() (int, bool) {
	if s == nil {
		return 0, false
	}
	config := s.config
	if config.TopK <= 0 || config.Mirostat != mirostatDisabled || config.NoRepeatNgramSize != 0 || config.Grammar != nil ||
		config.GBNF != nil || config.Infill != nil || len(config.LogitBiases) != 0 {
		return 0, false
	}
	for _, stage := range config.Samplers {
		switch stage {
		case SamplerPenalties:
			if config.RepeatPenalty != 1 || config.PresencePenalty != 0 || config.FrequencyPenalty != 0 {
				return 0, false
			}
		case SamplerDry:
			if config.DryMultiplier != 0 {
				return 0, false
			}
		case SamplerTopNSigma:
			if config.TopNSigma != 0 {
				return 0, false
			}
		case SamplerTypicalP:
			if config.TypicalP != 1 {
				return 0, false
			}
		case SamplerTopP:
			if config.TopP != 1 {
				return 0, false
			}
		case SamplerMinP:
			if config.MinP != 0 {
				return 0, false
			}
		case SamplerXTC:
			if config.XTCProbability != 0 {
				return 0, false
			}
		case SamplerTemperature:
			if config.Temperature != 1 || config.DynatempRange != 0 {
				return 0, false
			}
		case SamplerAdaptiveP, SamplerInfill:
			return 0, false
		case SamplerTopK:
			return config.TopK, true
		}
	}
	return 0, false
}

// Sample chooses one token; Temperature zero is exact greedy argmax with
// lowest token ID winning ties
func (s *Sampler) Sample(logits []float32) (int, error) {
	return s.SampleWithHistory(logits, nil)
}

// SampleWithHistory: applies repetition, presence, and frequency penalties over
// configured suffix of history before filtering and selection
func (s *Sampler) SampleWithHistory(logits []float32, history []int) (int, error) {
	if s == nil {
		return 0, errors.New("sampler is nil")
	}
	if len(logits) == 0 {
		return 0, errors.New("sampling logits are empty")
	}
	adjusted := s.copyLogits(logits)
	for index, value := range adjusted {
		if math.IsNaN(float64(value)) {
			return 0, fmt.Errorf("sampling logit %d is NaN", index)
		}
	}
	for index, bias := range s.config.LogitBiases {
		if bias.Token >= len(adjusted) {
			return 0, fmt.Errorf(
				"sampling logit bias %d token %d exceeds vocabulary size %d",
				index,
				bias.Token,
				len(adjusted),
			)
		}
		if math.IsInf(float64(bias.Bias), -1) {
			adjusted[bias.Token] = BannedLogit()
		} else {
			adjusted[bias.Token] += bias.Bias
		}
	}
	if err := s.applyGrammar(adjusted); err != nil {
		return 0, err
	}
	if err := applyNoRepeatNgram(adjusted, history, s.config.NoRepeatNgramSize, s.config.NgramWindow); err != nil {
		return 0, err
	}
	if s.IsRawGreedy() {
		// full candidate pipeline reduces to argmax; skip vocab-wide sort
		best := 0
		for index, value := range adjusted {
			if value > adjusted[best] {
				best = index
			}
		}
		s.recordGreedyProbability(best)
		return s.acceptGrammar(best)
	}
	if s.config.Mirostat != mirostatDisabled && s.config.Temperature == 0 {
		best := 0
		for index, value := range adjusted {
			if value > adjusted[best] {
				best = index
			}
		}
		s.recordGreedyProbability(best)
		return s.acceptGrammar(best)
	}
	if s.config.Mirostat == mirostatV1 {
		selected, err := s.sampleMirostatV1(adjusted)
		if err != nil {
			return 0, err
		}
		return s.acceptGrammar(selected)
	}
	if s.config.Mirostat == mirostatV2 {
		selected, err := s.sampleMirostatV2(adjusted)
		if err != nil {
			return 0, err
		}
		return s.acceptGrammar(selected)
	}

	candidates := s.candidates(len(adjusted))
	for index, value := range adjusted {
		candidates[index] = candidate{id: index, scaledLogit: float64(value)}
	}
	return s.sampleCandidatePipeline(candidates, len(logits), history, s.config.Samplers)
}

func applyNoRepeatNgram(logits []float32, history []int, size, window int) error {
	if size <= 0 || len(history) < size {
		return nil
	}
	start := 0
	if window > 0 && len(history) > window {
		start = len(history) - window
	}
	end := len(history) - size + 1
	if end <= start {
		return nil
	}
	prefix := history[len(history)-size+1:]
	for index := start; index < end; index++ {
		if size > 1 && !slices.Equal(history[index:index+size-1], prefix) {
			continue
		}
		token := history[index+size-1]
		if token < 0 || token >= len(logits) {
			return fmt.Errorf("sampling no-repeat token %d exceeds vocabulary size %d", token, len(logits))
		}
		logits[token] = BannedLogit()
	}
	return nil
}

// SampleTopK: complete descending top-K prefix.
func (s *Sampler) SampleTopK(ids []int, logits []float32, vocabularySize int) (int, error) {
	if s == nil {
		return 0, errors.New("sampler is nil")
	}
	limit, ok := s.BoundedTopK()
	if !ok {
		return 0, errors.New("sampling configuration has no bounded top-K prefix")
	}
	expected := min(limit, vocabularySize)
	if vocabularySize <= 0 || len(ids) != expected || len(logits) != expected {
		return 0, errors.New("sampling top-K candidates are incomplete")
	}
	s.topKSeen = resetScratchMap(s.topKSeen, len(ids))
	candidates := s.candidates(len(ids))
	for index, id := range ids {
		if id < 0 || id >= vocabularySize {
			return 0, fmt.Errorf("sampling top-K token %d exceeds vocabulary", id)
		}
		if _, duplicate := s.topKSeen[id]; duplicate {
			return 0, fmt.Errorf("sampling top-K token %d is duplicated", id)
		}
		s.topKSeen[id] = struct{}{}
		if math.IsNaN(float64(logits[index])) {
			return 0, fmt.Errorf("sampling top-K logit %d is NaN", index)
		}
		candidates[index] = candidate{id: id, scaledLogit: float64(logits[index])}
	}
	stages := s.config.Samplers
	for index, stage := range stages {
		if stage == SamplerTopK {
			return s.sampleCandidatePipeline(candidates, vocabularySize, nil, stages[index+1:])
		}
	}
	return 0, errors.New("sampling top-K stage is missing")
}

func (s *Sampler) sampleCandidatePipeline(
	candidates []candidate,
	vocabularySize int,
	history []int,
	stages []SamplerStage,
) (int, error) {
	var err error
	useAdaptive := false
	for _, stage := range stages {
		if stage == SamplerAdaptiveP {
			useAdaptive = true
			continue
		}
		candidates, err = s.applySamplerStage(candidates, vocabularySize, history, stage)
		if err != nil {
			return 0, err
		}
		if len(candidates) == 0 {
			return 0, fmt.Errorf("sampling sampler %q removed every candidate", stage)
		}
	}
	if useAdaptive {
		selected, originalProbability, sampleErr := s.sampleAdaptiveP(candidates, vocabularySize)
		if sampleErr != nil {
			return 0, sampleErr
		}
		token, acceptErr := s.acceptGrammar(selected)
		if acceptErr != nil {
			return 0, acceptErr
		}
		if s.config.AdaptiveTarget >= 0 {
			s.adaptiveSum = originalProbability +
				float64(s.config.AdaptiveDecay)*s.adaptiveSum
			s.adaptiveWeight = 1 +
				float64(s.config.AdaptiveDecay)*s.adaptiveWeight
		}
		total, probabilityErr := candidateProbabilities(candidates)
		if probabilityErr != nil {
			return 0, probabilityErr
		}
		s.recordCandidateProbabilities(candidates, total, token)
		return token, nil
	}
	total, err := candidateProbabilities(candidates)
	if err != nil {
		return 0, err
	}
	selected := sampleCandidateIndex(s.random.Float64(), candidates, total)
	selectedID := candidates[selected].id
	s.recordCandidateProbabilities(candidates, total, selectedID)
	return s.acceptGrammar(selectedID)
}

func (s *Sampler) SampleWithHistoryProbabilities(
	logits []float32,
	history []int,
	limit int,
) (SampleProbabilityResult, error) {
	if limit <= 0 {
		return SampleProbabilityResult{}, errors.New(
			"sampling probability limit must be positive",
		)
	}
	s.probabilityLimit = limit
	s.lastProbability = SampleProbabilityResult{}
	defer func() {
		s.probabilityLimit = 0
	}()
	token, err := s.SampleWithHistory(logits, history)
	if err != nil {
		return SampleProbabilityResult{}, err
	}
	result := s.lastProbability
	result.Token = token
	result.Top = slices.Clone(result.Top)
	return result, nil
}

// AcceptToken: grammar-only external token commit.
func (s *Sampler) AcceptToken(token int) error {
	if s == nil {
		return errors.New("sampler is nil")
	}
	_, err := s.acceptGrammar(token)
	return err
}

func (s *Sampler) recordGreedyProbability(token int) {
	if s.probabilityLimit <= 0 {
		return
	}
	s.lastProbability = SampleProbabilityResult{
		Token:               token,
		SelectedProbability: 1,
		Top: []TokenProbability{{
			ID:          token,
			Probability: 1,
		}},
	}
}

func (s *Sampler) recordCandidateProbabilities(
	candidates []candidate,
	total float64,
	selected int,
) {
	if s.probabilityLimit <= 0 {
		return
	}
	slices.SortStableFunc(candidates, compareCandidateProbability)
	result := SampleProbabilityResult{Token: selected}
	for _, item := range candidates {
		probability := item.probability / total
		if item.id == selected {
			result.SelectedProbability = probability
		}
		if probability <= 0 || len(result.Top) >= s.probabilityLimit {
			continue
		}
		result.Top = append(result.Top, TokenProbability{
			ID:          item.id,
			Probability: probability,
		})
	}
	s.lastProbability = result
}

func (s *Sampler) sampleAdaptiveP(candidates []candidate, vocabularySize int) (int, float64, error) {
	original, total, err := s.adaptiveCandidates(candidates, vocabularySize)
	if err != nil {
		return 0, 0, err
	}
	selected := sampleCandidateIndex(s.random.Float64(), candidates, total)
	token := candidates[selected].id
	return token, original[token], nil
}

func (s *Sampler) adaptiveCandidates(candidates []candidate, vocabularySize int) ([]float64, float64, error) {
	total, err := candidateProbabilities(candidates)
	if err != nil {
		return nil, 0, err
	}
	s.adaptiveScratch = resizeScratch(s.adaptiveScratch, vocabularySize)
	original := s.adaptiveScratch
	for _, item := range candidates {
		original[item.id] = item.probability / total
	}
	if s.config.AdaptiveTarget >= 0 {
		target := math.Max(0, math.Min(1, float64(s.config.AdaptiveTarget)))
		adapted := target
		if s.adaptiveWeight != 0 {
			adapted = 2*target - s.adaptiveSum/s.adaptiveWeight
		}
		adapted = math.Max(0, math.Min(1, adapted))
		for index := range candidates {
			if math.IsInf(candidates[index].scaledLogit, -1) {
				continue
			}
			distance := math.Abs(
				(original[candidates[index].id] - adapted) /
					dynamicRangeWidth,
			)
			candidates[index].scaledLogit = adaptivePeakLogit -
				adaptiveSharpness*distance*distance/(1+distance)
		}
		total, err = candidateProbabilities(candidates)
		if err != nil {
			return nil, 0, err
		}
	}
	return original, total, nil
}

func (s *Sampler) applySamplerStage(
	candidates []candidate,
	vocabularySize int,
	history []int,
	stage SamplerStage,
) ([]candidate, error) {
	switch stage {
	case SamplerPenalties:
		values := s.expandCandidateLogits(candidates, vocabularySize)
		if err := s.applyPenalties(values, history); err != nil {
			return nil, err
		}
		updateCandidateLogits(candidates, values)
	case SamplerDry:
		values := s.expandCandidateLogits(candidates, vocabularySize)
		s.applyDry(values, history)
		updateCandidateLogits(candidates, values)
	case SamplerTopNSigma:
		if s.config.TopNSigma > 0 && len(candidates) > 1 {
			maximum := candidates[0].scaledLogit
			var sum float64
			valid := 0
			for _, item := range candidates {
				if !math.IsInf(item.scaledLogit, -1) {
					maximum = math.Max(maximum, item.scaledLogit)
					sum += item.scaledLogit
					valid++
				}
			}
			mean := float64(0)
			if valid > 0 {
				mean = sum / float64(valid)
			}
			var variance float64
			for _, item := range candidates {
				if !math.IsInf(item.scaledLogit, -1) {
					delta := item.scaledLogit - mean
					variance += delta * delta
				}
			}
			deviation := float64(0)
			if valid > 0 {
				deviation = math.Sqrt(variance / float64(valid))
			}
			threshold := maximum - float64(s.config.TopNSigma)*deviation
			for index := range candidates {
				if candidates[index].scaledLogit < threshold {
					candidates[index].scaledLogit = math.Inf(-1)
				}
			}
		}
	case SamplerTopK:
		sortCandidates(candidates)
		if s.config.TopK > 0 && len(candidates) > s.config.TopK {
			candidates = candidates[:s.config.TopK]
		}
	case SamplerTypicalP:
		if s.config.TypicalP < 1 && len(candidates) > 1 {
			total, err := candidateProbabilities(candidates)
			if err != nil {
				return nil, err
			}
			var entropy float64
			for _, item := range candidates {
				probability := item.probability / total
				if probability > 0 {
					entropy -= probability * math.Log(probability)
				}
			}
			slices.SortStableFunc(candidates, func(leftCandidate, rightCandidate candidate) int {
				left := math.Abs(-math.Log(leftCandidate.probability/total) - entropy)
				right := math.Abs(-math.Log(rightCandidate.probability/total) - entropy)
				if left == right {
					return leftCandidate.id - rightCandidate.id
				}
				if left < right {
					return -1
				}
				return 1
			})
			var cumulative float64
			keep := 0
			for index := range candidates {
				cumulative += candidates[index].probability / total
				keep = index + 1
				if cumulative > float64(s.config.TypicalP) &&
					(s.config.MinKeep == 0 || keep >= s.config.MinKeep) {
					break
				}
			}
			candidates = candidates[:keep]
			sortCandidates(candidates)
		}
	case SamplerTopP:
		if s.config.TopP < 1 {
			sortCandidates(candidates)
			total, err := candidateProbabilities(candidates)
			if err != nil {
				return nil, err
			}
			var cumulative float64
			keep := 0
			for index := range candidates {
				cumulative += candidates[index].probability
				keep = index + 1
				if cumulative >= float64(s.config.TopP)*total &&
					keep >= s.config.MinKeep {
					break
				}
			}
			candidates = candidates[:keep]
		}
	case SamplerMinP:
		if s.config.MinP > 0 {
			sortCandidates(candidates)
			if _, err := candidateProbabilities(candidates); err != nil {
				return nil, err
			}
			threshold := candidates[0].probability * float64(s.config.MinP)
			count := 1
			for count < len(candidates) {
				if candidates[count].probability < threshold &&
					count >= s.config.MinKeep {
					break
				}
				count++
			}
			candidates = candidates[:count]
		}
	case SamplerXTC:
		if s.config.XTCProbability > 0 &&
			s.config.XTCThreshold <= xtcDisableThreshold &&
			len(candidates) >= 2 &&
			s.random.Float32() <= s.config.XTCProbability {
			sortCandidates(candidates)
			total, err := candidateProbabilities(candidates)
			if err != nil {
				return nil, err
			}
			posLast := 0
			for index, item := range candidates {
				if item.probability/total >= float64(s.config.XTCThreshold) {
					posLast = index
				} else {
					break
				}
			}
			if len(candidates)-posLast >= s.config.MinKeep && posLast > 0 {
				candidates = candidates[posLast:]
			}
		}
	case SamplerTemperature:
		temperature := float64(s.config.Temperature)
		if s.config.DynatempRange > 0 && len(candidates) > 1 {
			total, err := candidateProbabilities(candidates)
			if err != nil {
				return nil, err
			}
			var entropy float64
			for _, item := range candidates {
				probability := item.probability / total
				if probability > 0 {
					entropy -= probability * math.Log(probability)
				}
			}
			normalizedEntropy := entropy / math.Log(float64(len(candidates)))
			minimum := math.Max(
				0,
				float64(s.config.Temperature-s.config.DynatempRange),
			)
			maximum := float64(s.config.Temperature + s.config.DynatempRange)
			temperature = minimum +
				(maximum-minimum)*
					math.Pow(normalizedEntropy, float64(s.config.DynatempExponent))
		}
		if temperature <= 0 {
			sortCandidates(candidates)
			candidates = candidates[:1]
		} else {
			inverse := 1 / temperature
			for index := range candidates {
				candidates[index].scaledLogit *= inverse
			}
		}
	case SamplerInfill:
		var err error
		candidates, err = s.applyInfill(candidates, s.config.Infill)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("sampling unsupported sampler %q", stage)
	}
	return candidates, nil
}

func (s *Sampler) applyInfill(
	candidates []candidate,
	vocabulary *InfillVocabulary,
) ([]candidate, error) {
	if err := validateInfillVocabulary(vocabulary); err != nil {
		return nil, err
	}
	sortCandidates(candidates)
	total, err := candidateProbabilities(candidates)
	if err != nil {
		return nil, err
	}
	var textProbability, eogProbability float64
	for index := range candidates {
		candidates[index].probability /= total
		if vocabulary.EOG[candidates[index].id] {
			eogProbability += candidates[index].probability
		} else {
			textProbability += candidates[index].probability
		}
	}
	if 3*eogProbability*float64(len(candidates)) > textProbability {
		result := s.infillCandidates(0, len(candidates))
		for _, item := range candidates {
			if vocabulary.EOG[item.id] {
				result = append(result, item)
			}
		}
		return probabilitiesAsLogits(result)
	}
	for left := 0; left < len(candidates); left++ {
		if math.IsInf(candidates[left].scaledLogit, -1) {
			continue
		}
		leftPiece := vocabulary.Pieces[candidates[left].id]
		if leftPiece == "" {
			continue
		}
		for right := 0; right < len(candidates); right++ {
			if left == right ||
				math.IsInf(candidates[right].scaledLogit, -1) {
				continue
			}
			rightPiece := vocabulary.Pieces[candidates[right].id]
			if len(leftPiece) > len(rightPiece) ||
				!strings.HasPrefix(rightPiece, leftPiece) {
				continue
			}
			destination, source := left, right
			if candidates[right].probability >
				candidates[left].probability {
				destination, source = right, left
			}
			candidates[destination].probability +=
				candidates[source].probability
			candidates[source].scaledLogit = math.Inf(-1)
			candidates[source].probability = 0
			if source == left {
				break
			}
		}
	}
	firstPass := s.infillCandidates(0, len(candidates))
	nonEOG := 0
	for _, item := range candidates {
		isEOG := vocabulary.EOG[item.id]
		if item.probability < adaptiveTailCutoff && !isEOG {
			continue
		}
		if !isEOG {
			nonEOG++
		}
		firstPass = append(firstPass, item)
	}
	if nonEOG == 0 {
		token := vocabulary.EOT
		if token < 0 {
			token = vocabulary.EOS
		}
		if token < 0 {
			return nil, errors.New("infill vocabulary has no EOT or EOS token")
		}
		return []candidate{{
			id:          token,
			scaledLogit: 0,
			probability: 1,
		}}, nil
	}
	firstPass, err = normalizeCandidateProbabilities(firstPass)
	if err != nil {
		return nil, err
	}
	threshold := 1 / float64(nonEOG+1)
	secondPass := s.infillCandidates(1, len(firstPass))
	for _, item := range firstPass {
		if item.probability < threshold &&
			!vocabulary.EOG[item.id] {
			continue
		}
		secondPass = append(secondPass, item)
	}
	return probabilitiesAsLogits(secondPass)
}

func normalizeCandidateProbabilities(
	candidates []candidate,
) ([]candidate, error) {
	var total float64
	for _, item := range candidates {
		total += item.probability
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return nil, errors.New("sampling infill probability normalization failed")
	}
	for index := range candidates {
		candidates[index].probability /= total
	}
	return candidates, nil
}

func probabilitiesAsLogits(
	candidates []candidate,
) ([]candidate, error) {
	candidates, err := normalizeCandidateProbabilities(candidates)
	if err != nil {
		return nil, err
	}
	for index := range candidates {
		if candidates[index].probability == 0 {
			candidates[index].scaledLogit = math.Inf(-1)
		} else {
			candidates[index].scaledLogit =
				math.Log(candidates[index].probability)
		}
	}
	return candidates, nil
}

func validateInfillVocabulary(vocabulary *InfillVocabulary) error {
	if vocabulary == nil {
		return errors.New("vocabulary is nil")
	}
	if len(vocabulary.Pieces) == 0 ||
		len(vocabulary.EOG) != len(vocabulary.Pieces) {
		return errors.New("piece and EOG tables must have equal nonzero lengths")
	}
	for name, token := range map[string]int{
		"EOT": vocabulary.EOT,
		"EOS": vocabulary.EOS,
	} {
		if token < -1 || token >= len(vocabulary.Pieces) {
			return fmt.Errorf("%s token %d is out of range", name, token)
		}
	}
	return nil
}

func cloneInfillVocabulary(
	vocabulary *InfillVocabulary,
) *InfillVocabulary {
	if vocabulary == nil {
		return nil
	}
	return &InfillVocabulary{
		Pieces:    slices.Clone(vocabulary.Pieces),
		EOG:       slices.Clone(vocabulary.EOG),
		EOT:       vocabulary.EOT,
		EOS:       vocabulary.EOS,
		signature: vocabulary.signature,
	}
}

func (s *Sampler) expandCandidateLogits(candidates []candidate, vocabularySize int) []float32 {
	if cap(s.logitScratch) < vocabularySize {
		s.logitScratch = make([]float32, vocabularySize)
	} else {
		s.logitScratch = s.logitScratch[:vocabularySize]
	}
	result := s.logitScratch
	for index := range result {
		result[index] = float32(math.Inf(-1))
	}
	for _, item := range candidates {
		result[item.id] = float32(item.scaledLogit)
	}
	return result
}

func (s *Sampler) copyLogits(logits []float32) []float32 {
	s.adjustedScratch = append(s.adjustedScratch[:0], logits...)
	return s.adjustedScratch
}

func (s *Sampler) candidates(count int) []candidate {
	if cap(s.candidateScratch) < count {
		s.candidateScratch = make([]candidate, count)
	} else {
		s.candidateScratch = s.candidateScratch[:count]
		clear(s.candidateScratch)
	}
	return s.candidateScratch
}

func (s *Sampler) infillCandidates(index, capacity int) []candidate {
	result := s.infillScratch[index][:0]
	if cap(result) < capacity {
		result = make([]candidate, 0, capacity)
	}
	s.infillScratch[index] = result
	return result
}

func resizeScratch[T any](scratch []T, count int) []T {
	if cap(scratch) < count {
		return make([]T, count)
	}
	scratch = scratch[:count]
	clear(scratch)
	return scratch
}

func resetScratchMap[K comparable, V any](scratch map[K]V, capacity int) map[K]V {
	if scratch == nil {
		return make(map[K]V, capacity)
	}
	clear(scratch)
	return scratch
}

func updateCandidateLogits(candidates []candidate, logits []float32) {
	for index := range candidates {
		candidates[index].scaledLogit = float64(logits[candidates[index].id])
	}
}

func sortCandidates(candidates []candidate) {
	slices.SortFunc(candidates, compareCandidateLogit)
}

func compareCandidateLogit(left, right candidate) int {
	if left.scaledLogit == right.scaledLogit {
		return left.id - right.id
	}
	if left.scaledLogit > right.scaledLogit {
		return -1
	}
	return 1
}

func compareCandidateProbability(left, right candidate) int {
	if left.probability == right.probability {
		return left.id - right.id
	}
	if left.probability > right.probability {
		return -1
	}
	return 1
}

func candidateProbabilities(candidates []candidate) (float64, error) {
	maximum := candidates[0].scaledLogit
	for _, item := range candidates[1:] {
		if item.scaledLogit > maximum {
			maximum = item.scaledLogit
		}
	}
	var total float64
	for index := range candidates {
		candidates[index].probability = math.Exp(candidates[index].scaledLogit - maximum)
		total += candidates[index].probability
	}
	if total == 0 || math.IsInf(total, 0) || math.IsNaN(total) {
		return 0, errors.New("sampling probability normalization failed")
	}
	return total, nil
}

// Reset: restores adaptive and random state to initial configuration
func (s *Sampler) Reset() {
	if s == nil {
		return
	}
	s.source.Seed(s.config.Seed)
	s.random = rand.New(s.source)
	s.mu = 2 * float64(s.config.MirostatTau)
	s.probabilityLimit = 0
	s.lastProbability = SampleProbabilityResult{}
	s.resetAdaptive()
	if s.grammar != nil {
		s.grammarState = s.grammar.Start
	}
	if s.gbnf != nil {
		state, _ := s.gbnf.initialState()
		s.gbnfState = state
		s.gbnfHistory = nil
	}
}

func (s *Sampler) resetAdaptive() {
	if s == nil || !s.adaptive {
		s.adaptiveSum = 0
		s.adaptiveWeight = 0
		return
	}
	denominator := 1 - float64(s.config.AdaptiveDecay)
	s.adaptiveSum = float64(s.config.AdaptiveTarget) / denominator
	s.adaptiveWeight = 1 / denominator
}

func (s *Sampler) applyGrammar(logits []float32) error {
	if s.gbnf != nil {
		if s.gbnfState.awaitingTrigger {
			return nil
		}
		if len(s.gbnf.tokenPieces) != len(logits) {
			return fmt.Errorf(
				"sampling GBNF vocabulary has %d tokens, logits have %d",
				len(s.gbnf.tokenPieces),
				len(logits),
			)
		}
		allowed := 0
		for token := range logits {
			if _, ok := s.gbnf.advanceToken(s.gbnfState, token); !ok {
				logits[token] = float32(math.Inf(-1))
			} else {
				allowed++
			}
		}
		if allowed == 0 {
			return errors.New("sampling GBNF has no valid next token")
		}
		return nil
	}
	if s.grammar == nil {
		return nil
	}
	if s.grammarState < 0 || s.grammarState >= len(s.grammar.Transitions) {
		return errors.New("sampling grammar state is invalid")
	}
	transitions := s.grammar.Transitions[s.grammarState]
	if s.grammar.VocabularySize != len(logits) {
		return fmt.Errorf(
			"sampling grammar vocabulary has %d tokens, logits have %d",
			s.grammar.VocabularySize,
			len(logits),
		)
	}
	if len(transitions) == 0 {
		return errors.New("sampling grammar has no valid next token")
	}
	for token := range logits {
		if _, ok := transitions[token]; !ok {
			logits[token] = float32(math.Inf(-1))
		}
	}
	return nil
}

func (s *Sampler) acceptGrammar(token int) (int, error) {
	if s.gbnf != nil {
		next, ok := s.gbnf.advanceToken(s.gbnfState, token)
		if !ok {
			return 0, errors.New("sampling selected a token rejected by GBNF")
		}
		s.gbnfState = next
		s.gbnfHistory = append(s.gbnfHistory, token)
		return token, nil
	}
	if s.grammar == nil {
		return token, nil
	}
	transitions := s.grammar.Transitions[s.grammarState]
	next, ok := transitions[token]
	if token < 0 || token >= s.grammar.VocabularySize || !ok {
		return 0, errors.New("sampling selected a token rejected by grammar")
	}
	s.grammarState = next
	return token, nil
}

func (s *Sampler) sampleMirostatV2(logits []float32) (int, error) {
	candidates := s.candidates(len(logits))
	inverseTemperature := 1 / float64(s.config.Temperature)
	for index, value := range logits {
		candidates[index] = candidate{
			id:          index,
			scaledLogit: float64(value) * inverseTemperature,
		}
	}
	slices.SortFunc(candidates, compareCandidateLogit)
	maximum := candidates[0].scaledLogit
	var total float64
	for index := range candidates {
		candidates[index].probability = math.Exp(candidates[index].scaledLogit - maximum)
		total += candidates[index].probability
	}
	if total == 0 || math.IsInf(total, 0) || math.IsNaN(total) {
		return 0, errors.New("Mirostat probability normalization failed")
	}
	keep := 0
	for index := range candidates {
		candidates[index].probability /= total
		surprise := -math.Log2(candidates[index].probability)
		if surprise > s.mu {
			break
		}
		keep = index + 1
	}
	if keep == 0 {
		keep = 1
	}
	candidates = candidates[:keep]
	total = 0
	for index := range candidates {
		total += candidates[index].probability
	}
	target := s.random.Float64() * total
	var cumulative float64
	selected := len(candidates) - 1
	for index, item := range candidates {
		cumulative += item.probability
		if target < cumulative {
			selected = index
			break
		}
	}
	normalizedProbability := candidates[selected].probability / total
	observedSurprise := -math.Log2(normalizedProbability)
	errorValue := observedSurprise - float64(s.config.MirostatTau)
	s.mu -= float64(s.config.MirostatEta) * errorValue
	selectedID := candidates[selected].id
	s.recordCandidateProbabilities(candidates, total, selectedID)
	return selectedID, nil
}

func (s *Sampler) sampleMirostatV1(logits []float32) (int, error) {
	candidates, total, err := normalizedCandidatesInto(
		s.candidates(len(logits)), logits, s.config.Temperature,
	)
	if err != nil {
		return 0, fmt.Errorf("Mirostat v1: %w", err)
	}
	keep := mirostatV1CandidateCount(candidates, total, len(logits), s.mu)
	candidates = candidates[:keep]
	total = 0
	for _, item := range candidates {
		total += item.probability
	}
	selected := sampleCandidateIndex(s.random.Float64(), candidates, total)
	normalizedProbability := candidates[selected].probability / total
	observedSurprise := -math.Log2(normalizedProbability)
	errorValue := observedSurprise - float64(s.config.MirostatTau)
	s.mu -= float64(s.config.MirostatEta) * errorValue
	selectedID := candidates[selected].id
	s.recordCandidateProbabilities(candidates, total, selectedID)
	return selectedID, nil
}

const mirostatV1EstimateTokens = 100

func mirostatV1CandidateCount(candidates []candidate, total float64, vocabulary int, mu float64) int {
	estimateCount := min(mirostatV1EstimateTokens-1, len(candidates)-1)
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
	if sumTISquared == 0 {
		return 1
	}
	sHat := sumTIBI / sumTISquared
	epsilonHat := sHat - 1
	denominator := 1 - math.Pow(float64(vocabulary), -epsilonHat)
	k := math.Pow(epsilonHat*math.Exp2(mu)/denominator, 1/sHat)
	if math.IsNaN(k) || math.IsInf(k, 0) || k <= 1 {
		return 1
	}
	if k >= float64(len(candidates)) {
		return len(candidates)
	}
	return int(k)
}

func normalizedCandidatesInto(candidates []candidate, logits []float32, temperature float32) ([]candidate, float64, error) {
	if len(candidates) != len(logits) {
		return nil, 0, errors.New("candidate scratch size differs from logits")
	}
	inverseTemperature := 1 / float64(temperature)
	for index, value := range logits {
		candidates[index] = candidate{
			id:          index,
			scaledLogit: float64(value) * inverseTemperature,
		}
	}
	slices.SortFunc(candidates, compareCandidateLogit)
	maximum := candidates[0].scaledLogit
	var total float64
	for index := range candidates {
		candidates[index].probability = math.Exp(candidates[index].scaledLogit - maximum)
		total += candidates[index].probability
	}
	if total == 0 || math.IsInf(total, 0) || math.IsNaN(total) {
		return nil, 0, errors.New("probability normalization failed")
	}
	return candidates, total, nil
}

func sampleCandidateIndex(random float64, candidates []candidate, total float64) int {
	target := random * total
	var cumulative float64
	for index, item := range candidates {
		cumulative += item.probability
		if target < cumulative {
			return index
		}
	}
	return len(candidates) - 1
}

func (s *Sampler) applyPenalties(logits []float32, history []int) error {
	if len(history) == 0 || s.config.RepeatLastN == 0 ||
		(s.config.RepeatPenalty == 1 &&
			s.config.PresencePenalty == 0 &&
			s.config.FrequencyPenalty == 0) {
		return nil
	}
	first := 0
	if s.config.RepeatLastN >= 0 && len(history) > s.config.RepeatLastN {
		first = len(history) - s.config.RepeatLastN
	}
	s.penaltyCounts = resetScratchMap(s.penaltyCounts, len(history)-first)
	for _, id := range history[first:] {
		if id >= 0 && id < len(logits) {
			s.penaltyCounts[id]++
		}
	}
	for id, count := range s.penaltyCounts {
		value := logits[id]
		if value <= 0 {
			value *= s.config.RepeatPenalty
		} else {
			value /= s.config.RepeatPenalty
		}
		value -= s.config.PresencePenalty
		value -= float32(count) * s.config.FrequencyPenalty
		logits[id] = value
	}
	return nil
}

// applyDry ports llama.cpp's reverse Z-algorithm for finding suffixes that
// would be extended by each candidate token; Sequence breakers are added at
// higher layer later; implements core token-history penalty
func (s *Sampler) applyDry(logits []float32, history []int) {
	if s.config.DryMultiplier == 0 || s.config.DryPenaltyLastN == 0 ||
		len(history) <= s.config.DryAllowedLength {
		return
	}
	lastN := len(history)
	if s.config.DryPenaltyLastN >= 0 && lastN > s.config.DryPenaltyLastN {
		lastN = s.config.DryPenaltyLastN
	}
	if lastN <= s.config.DryAllowedLength {
		return
	}
	s.dryReversed = resizeScratch(s.dryReversed, lastN)
	reversed := s.dryReversed
	for index := range lastN {
		reversed[index] = history[len(history)-1-index]
	}
	repLimit := lastN
	for index, token := range reversed {
		longestMatch := -1
		for _, tail := range s.dryBreakers[token] {
			if len(tail) <= longestMatch || len(tail) > index {
				continue
			}
			matches := true
			for offset, tailToken := range tail {
				if tailToken != reversed[index-offset-1] {
					matches = false
					break
				}
			}
			if matches {
				longestMatch = len(tail)
			}
		}
		if longestMatch >= 0 {
			repLimit = index - longestMatch
			break
		}
	}
	if repLimit < s.config.DryAllowedLength {
		return
	}
	s.dryRepeatCount = resizeScratch(s.dryRepeatCount, lastN)
	repeatCount := s.dryRepeatCount
	last := lastN - 1
	right, left := 0, 0
	for k := 1; k < lastN; k++ {
		if k > right {
			count := 0
			for count+k < lastN && reversed[count] == reversed[count+k] {
				count++
			}
			repeatCount[last-k] = min(count, repLimit)
			if count > 0 {
				left = k
				right = k + count - 1
			}
		} else {
			pair := k - left
			rightLength := right - k + 1
			if repeatCount[last-pair] < rightLength {
				repeatCount[last-k] = min(repeatCount[last-pair], repLimit)
			} else {
				index := right + 1
				for index < lastN && reversed[index] == reversed[index-k] {
					index++
				}
				repeatCount[last-k] = min(index-k, repLimit)
				left = k
				right = index - 1
			}
		}
	}
	s.dryMaxRepeat = resetScratchMap(s.dryMaxRepeat, lastN)
	maxRepeat := s.dryMaxRepeat
	for index := 0; index < lastN-1; index++ {
		repeatLength := repeatCount[index]
		if repeatLength < s.config.DryAllowedLength {
			continue
		}
		token := reversed[lastN-2-index]
		if token < 0 || token >= len(logits) {
			continue
		}
		if repeatLength > maxRepeat[token] {
			maxRepeat[token] = repeatLength
		}
	}
	maxExponent := 0
	if s.config.DryBase > dryBaseTolerance {
		maxExponent = int(math.Log(math.MaxFloat32) / math.Log(float64(s.config.DryBase)))
	}
	for token, repeatLength := range maxRepeat {
		singleTokenBreaker := false
		for _, tail := range s.dryBreakers[token] {
			if len(tail) == 0 {
				singleTokenBreaker = true
				break
			}
		}
		if singleTokenBreaker {
			continue
		}
		exponent := repeatLength - s.config.DryAllowedLength
		if maxExponent > 0 && exponent > maxExponent {
			exponent = maxExponent
		}
		penalty := float64(s.config.DryMultiplier) *
			math.Pow(float64(s.config.DryBase), float64(exponent))
		logits[token] -= float32(penalty)
	}
}

func cloneBreakers(breakers [][]int) [][]int {
	if breakers == nil {
		return nil
	}
	result := make([][]int, len(breakers))
	for index, breaker := range breakers {
		result[index] = slices.Clone(breaker)
	}
	return result
}
