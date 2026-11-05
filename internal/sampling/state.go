package sampling

import (
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/statecodec"
)

const (
	samplerStateMagic      = "L2GSMP04"
	samplerStateMagicBytes = uint64(len(samplerStateMagic))
	maxSamplerStateBytes   = 1 << 20
	// emptyChainSignature parts an explicitly empty chain from the omitted order.
	emptyChainSignature = ^uint64(8)
)

// SaveState: stores random stream, Mirostat, adaptive-p, and grammar state
func (s *Sampler) SaveState() ([]byte, error) {
	if s == nil || s.source == nil {
		return nil, errors.New("sampler is nil")
	}
	encoder := statecodec.NewEncoder(maxSamplerStateBytes)
	encoder.Raw([]byte(samplerStateMagic))
	encoder.U64(configSignature(s.config))
	encoder.U64(s.source.state)
	encoder.F64(s.mu)
	encoder.U64(uint64(s.grammarState))
	encoder.U32(uint32(len(s.gbnfHistory)))
	encoder.F64(s.adaptiveSum)
	encoder.F64(s.adaptiveWeight)
	if uint64(len(s.gbnfHistory)) > encoder.Remaining()/binaryschema.Uint32Bytes {
		return nil, errors.New("sampler GBNF history exceeds state limit")
	}
	for _, token := range s.gbnfHistory {
		encoder.U32(uint32(token))
	}
	return encoder.Data()
}

// LoadState: restores state only when it was produced by identical sampler
// configuration
func (s *Sampler) LoadState(data []byte) error {
	if s == nil || s.source == nil {
		return errors.New("sampler is nil")
	}
	if len(data) == 0 || len(data) > maxSamplerStateBytes {
		return errors.New("sampler state has invalid size")
	}
	decoder := statecodec.NewDecoder(data, maxSamplerStateBytes)
	magic := string(decoder.Raw(samplerStateMagicBytes))
	if magic != samplerStateMagic {
		return errors.New("sampler state has invalid magic or version")
	}
	signature := decoder.U64()
	sourceState := decoder.U64()
	mu := decoder.F64()
	grammarState := decoder.U64()
	count := decoder.U32()
	adaptiveSum := decoder.F64()
	adaptiveWeight := decoder.F64()
	if decoder.Err() != nil {
		return errors.New("sampler state has invalid size")
	}
	if signature != configSignature(s.config) {
		return errors.New("sampler state configuration does not match")
	}
	if math.IsNaN(mu) || math.IsInf(mu, 0) {
		return errors.New("sampler state has invalid Mirostat value")
	}
	nextGrammarState := 0
	if s.grammar == nil {
		if grammarState != 0 {
			return errors.New("sampler state has unexpected grammar state")
		}
	} else {
		if grammarState >= uint64(len(s.grammar.Transitions)) {
			return errors.New("sampler state has invalid grammar state")
		}
		nextGrammarState = int(grammarState)
	}
	var nextGBNFState gbnfState
	var nextGBNFHistory []int
	nextAdaptiveSum := 0.0
	nextAdaptiveWeight := 0.0
	if s.adaptive {
		nextAdaptiveSum = adaptiveSum
		nextAdaptiveWeight = adaptiveWeight
		if math.IsNaN(nextAdaptiveSum) || math.IsInf(nextAdaptiveSum, 0) ||
			math.IsNaN(nextAdaptiveWeight) || math.IsInf(nextAdaptiveWeight, 0) ||
			nextAdaptiveWeight <= 0 {
			return errors.New("sampler state has invalid adaptive-p state")
		}
	} else {
		if adaptiveSum != 0 || adaptiveWeight != 0 {
			return errors.New("sampler state has unexpected adaptive-p state")
		}
	}
	if s.gbnf != nil {
		var err error
		nextGBNFState, err = s.gbnf.initialState()
		if err != nil {
			return fmt.Errorf("sampler state initialize GBNF: %w", err)
		}
	}
	if uint64(count) > decoder.Remaining()/binaryschema.Uint32Bytes {
		return errors.New("sampler state has invalid GBNF history length")
	}
	if s.gbnf == nil && count != 0 {
		return errors.New("sampler state has unexpected GBNF history")
	}
	nextGBNFHistory = make([]int, int(count))
	for index := range nextGBNFHistory {
		token := decoder.U32()
		if token > math.MaxInt32 {
			return errors.New("sampler state has invalid GBNF token")
		}
		nextGBNFHistory[index] = int(token)
		var ok bool
		nextGBNFState, ok = s.gbnf.advanceToken(nextGBNFState, int(token))
		if !ok {
			return fmt.Errorf(
				"sampler state GBNF history token %d is rejected",
				index,
			)
		}
	}
	if err := decoder.Done(); err != nil {
		return errors.New("sampler state has invalid GBNF history length")
	}
	s.source.state = sourceState
	s.mu = mu
	s.grammarState = nextGrammarState
	s.gbnfState = nextGBNFState
	s.gbnfHistory = nextGBNFHistory
	s.adaptiveSum = nextAdaptiveSum
	s.adaptiveWeight = nextAdaptiveWeight
	return nil
}

func configSignature(config Config) uint64 {
	hash := fnv.New64a()
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.Temperature)))
	binaryschema.HashUint64(hash, uint64(config.TopK))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.TopP)))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.MinP)))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.TypicalP)))
	binaryschema.HashUint64(hash, uint64(config.RepeatLastN))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.RepeatPenalty)))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.PresencePenalty)))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.FrequencyPenalty)))
	if config.NoRepeatNgramSize > 0 {
		binaryschema.HashUint64(hash, ^uint64(7))
		binaryschema.HashUint64(hash, uint64(config.NoRepeatNgramSize))
		binaryschema.HashUint64(hash, uint64(config.NgramWindow))
	}
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.DryMultiplier)))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.DryBase)))
	binaryschema.HashUint64(hash, uint64(config.DryAllowedLength))
	binaryschema.HashUint64(hash, uint64(config.DryPenaltyLastN))
	binaryschema.HashUint64(hash, uint64(len(config.DryBreakers)))
	for _, breaker := range config.DryBreakers {
		binaryschema.HashUint64(hash, uint64(len(breaker)))
		for _, token := range breaker {
			binaryschema.HashUint64(hash, uint64(token))
		}
	}
	binaryschema.HashUint64(hash, uint64(config.Mirostat))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.MirostatTau)))
	binaryschema.HashUint64(hash, uint64(math.Float32bits(config.MirostatEta)))
	binaryschema.HashUint64(hash, uint64(config.Seed))
	if config.Grammar == nil {
		binaryschema.HashUint64(hash, 0)
	} else {
		binaryschema.HashUint64(hash, uint64(config.Grammar.Start+1))
		binaryschema.HashUint64(hash, uint64(config.Grammar.VocabularySize))
		binaryschema.HashUint64(hash, uint64(len(config.Grammar.Transitions)))
		for state, transitions := range config.Grammar.Transitions {
			if config.Grammar.Accepting[state] {
				binaryschema.HashUint64(hash, 1)
			} else {
				binaryschema.HashUint64(hash, 0)
			}
			tokens := make([]int, 0, len(transitions))
			for token := range transitions {
				tokens = append(tokens, token)
			}
			slices.Sort(tokens)
			binaryschema.HashUint64(hash, uint64(len(tokens)))
			for _, token := range tokens {
				binaryschema.HashUint64(hash, uint64(token))
				binaryschema.HashUint64(hash, uint64(transitions[token]+1))
			}
		}
	}
	// Preserve v2 signature byte-for-byte when no GBNF is configured so
	// existing sampler/session states remain loadable
	if config.GBNF != nil {
		binaryschema.HashUint64(hash, ^uint64(0))
		binaryschema.HashUint64(hash, config.GBNF.signature)
	}
	if config.TopNSigma > 0 || config.XTCProbability > 0 || config.MinKeep > 0 {
		binaryschema.HashUint64(hash, ^uint64(2))
		binaryschema.HashUint64(hash, uint64(math.Float32bits(config.TopNSigma)))
		binaryschema.HashUint64(hash, uint64(math.Float32bits(config.XTCProbability)))
		binaryschema.HashUint64(hash, uint64(math.Float32bits(config.XTCThreshold)))
		binaryschema.HashUint64(hash, uint64(config.MinKeep))
	}
	if config.DynatempRange > 0 {
		binaryschema.HashUint64(hash, ^uint64(3))
		binaryschema.HashUint64(hash, uint64(math.Float32bits(config.DynatempRange)))
		binaryschema.HashUint64(hash, uint64(math.Float32bits(config.DynatempExponent)))
	}
	if slices.Contains(config.Samplers, SamplerAdaptiveP) {
		binaryschema.HashUint64(hash, ^uint64(4))
		binaryschema.HashUint64(hash, uint64(math.Float32bits(config.AdaptiveTarget)))
		binaryschema.HashUint64(hash, uint64(math.Float32bits(config.AdaptiveDecay)))
	}
	if len(config.LogitBiases) > 0 {
		binaryschema.HashUint64(hash, ^uint64(5))
		binaryschema.HashUint64(hash, uint64(len(config.LogitBiases)))
		for _, bias := range config.LogitBiases {
			binaryschema.HashUint64(hash, uint64(bias.Token))
			binaryschema.HashUint64(hash, uint64(math.Float32bits(bias.Bias)))
		}
	}
	if config.Infill != nil {
		binaryschema.HashUint64(hash, ^uint64(6))
		binaryschema.HashUint64(hash, config.Infill.signature)
	}
	if !slices.Equal(config.Samplers, defaultSamplerOrder) {
		binaryschema.HashUint64(hash, ^uint64(1))
		binaryschema.HashUint64(hash, uint64(len(config.Samplers)))
		for _, stage := range config.Samplers {
			binaryschema.HashUint64(hash, uint64(len(stage)))
			_, _ = hash.Write([]byte(stage))
		}
	}
	if config.Samplers != nil && len(config.Samplers) == 0 {
		binaryschema.HashUint64(hash, emptyChainSignature)
	}
	return hash.Sum64()
}

func infillVocabularySignature(vocabulary *InfillVocabulary) uint64 {
	hash := fnv.New64a()
	binaryschema.HashUint64(hash, uint64(len(vocabulary.Pieces)))
	for index, piece := range vocabulary.Pieces {
		binaryschema.HashUint64(hash, uint64(len(piece)))
		_, _ = hash.Write([]byte(piece))
		if vocabulary.EOG[index] {
			binaryschema.HashUint64(hash, 1)
		} else {
			binaryschema.HashUint64(hash, 0)
		}
	}
	binaryschema.HashUint64(hash, uint64(vocabulary.EOT+1))
	binaryschema.HashUint64(hash, uint64(vocabulary.EOS+1))
	return hash.Sum64()
}
