package sampling

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"sort"

	"overgo/internal/statecodec"
)

const (
	samplerStateMagic         = "L2GSMP04"
	previousSamplerStateMagic = "L2GSMP03"
	legacySamplerStateMagic   = "L2GSMP02"
	samplerStateMagicBytes    = uint64(len(samplerStateMagic))
	samplerStateHeaderSize    = 60
	previousSamplerHeaderSize = 44
	legacySamplerStateSize    = 40
	samplerTokenBytes         = 4
	maxSamplerStateBytes      = 1 << 20
	maxGBNFHistoryTokens      = (maxSamplerStateBytes - samplerStateHeaderSize) / samplerTokenBytes
)

// SaveState: stores random stream, Mirostat, adaptive-p, and grammar state
func (s *Sampler) SaveState() ([]byte, error) {
	if s == nil || s.source == nil {
		return nil, errors.New("sampler is nil")
	}
	if len(s.gbnfHistory) > maxGBNFHistoryTokens {
		return nil, errors.New("sampler GBNF history exceeds state limit")
	}
	size := uint64(samplerStateHeaderSize + len(s.gbnfHistory)*samplerTokenBytes)
	encoder := statecodec.NewEncoderCapacity(maxSamplerStateBytes, size)
	encoder.Raw([]byte(samplerStateMagic))
	encoder.U64(configSignature(s.config))
	encoder.U64(s.source.state)
	encoder.F64(s.mu)
	encoder.U64(uint64(s.grammarState))
	encoder.U32(uint32(len(s.gbnfHistory)))
	encoder.F64(s.adaptiveSum)
	encoder.F64(s.adaptiveWeight)
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
	decoder := statecodec.NewDecoder(data, maxSamplerStateBytes)
	magic := string(decoder.Raw(samplerStateMagicBytes))
	legacy := magic == legacySamplerStateMagic
	previous := magic == previousSamplerStateMagic
	if magic != samplerStateMagic && !previous && !legacy {
		return errors.New("sampler state has invalid magic or version")
	}
	if legacy {
		if len(data) != legacySamplerStateSize {
			return errors.New("sampler state has invalid size")
		}
		if s.gbnf != nil {
			return errors.New("legacy sampler state cannot restore GBNF history")
		}
	} else if previous {
		if len(data) < previousSamplerHeaderSize {
			return errors.New("sampler state has invalid size")
		}
		if s.adaptive {
			return errors.New("previous sampler state cannot restore adaptive-p state")
		}
	}
	signature := decoder.U64()
	sourceState := decoder.U64()
	mu := decoder.F64()
	grammarState := decoder.U64()
	count := uint32(0)
	if !legacy {
		count = decoder.U32()
	}
	adaptiveSum, adaptiveWeight := 0.0, 0.0
	if !legacy && !previous {
		adaptiveSum = decoder.F64()
		adaptiveWeight = decoder.F64()
	}
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
		if legacy || previous {
			return errors.New("legacy sampler state lacks adaptive-p state")
		}
		nextAdaptiveSum = adaptiveSum
		nextAdaptiveWeight = adaptiveWeight
		if math.IsNaN(nextAdaptiveSum) || math.IsInf(nextAdaptiveSum, 0) ||
			math.IsNaN(nextAdaptiveWeight) || math.IsInf(nextAdaptiveWeight, 0) ||
			nextAdaptiveWeight <= 0 {
			return errors.New("sampler state has invalid adaptive-p state")
		}
	} else if !legacy && !previous {
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
	if !legacy {
		maxHistory := maxGBNFHistoryTokens
		if previous {
			maxHistory = (maxSamplerStateBytes - previousSamplerHeaderSize) / samplerTokenBytes
		}
		if count > uint32(maxHistory) {
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
	} else if s.gbnf != nil {
		return errors.New("legacy sampler state lacks GBNF history")
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
	var buffer [8]byte
	writeUint64 := func(value uint64) {
		binary.LittleEndian.PutUint64(buffer[:], value)
		_, _ = hash.Write(buffer[:])
	}
	writeUint64(uint64(math.Float32bits(config.Temperature)))
	writeUint64(uint64(config.TopK))
	writeUint64(uint64(math.Float32bits(config.TopP)))
	writeUint64(uint64(math.Float32bits(config.MinP)))
	writeUint64(uint64(math.Float32bits(config.TypicalP)))
	writeUint64(uint64(config.RepeatLastN))
	writeUint64(uint64(math.Float32bits(config.RepeatPenalty)))
	writeUint64(uint64(math.Float32bits(config.PresencePenalty)))
	writeUint64(uint64(math.Float32bits(config.FrequencyPenalty)))
	writeUint64(uint64(math.Float32bits(config.DryMultiplier)))
	writeUint64(uint64(math.Float32bits(config.DryBase)))
	writeUint64(uint64(config.DryAllowedLength))
	writeUint64(uint64(config.DryPenaltyLastN))
	writeUint64(uint64(len(config.DryBreakers)))
	for _, breaker := range config.DryBreakers {
		writeUint64(uint64(len(breaker)))
		for _, token := range breaker {
			writeUint64(uint64(token))
		}
	}
	writeUint64(uint64(config.Mirostat))
	writeUint64(uint64(math.Float32bits(config.MirostatTau)))
	writeUint64(uint64(math.Float32bits(config.MirostatEta)))
	writeUint64(uint64(config.Seed))
	if config.Grammar == nil {
		writeUint64(0)
	} else {
		writeUint64(uint64(config.Grammar.Start + 1))
		writeUint64(uint64(config.Grammar.VocabularySize))
		writeUint64(uint64(len(config.Grammar.Transitions)))
		for state, transitions := range config.Grammar.Transitions {
			if config.Grammar.Accepting[state] {
				writeUint64(1)
			} else {
				writeUint64(0)
			}
			tokens := make([]int, 0, len(transitions))
			for token := range transitions {
				tokens = append(tokens, token)
			}
			sort.Ints(tokens)
			writeUint64(uint64(len(tokens)))
			for _, token := range tokens {
				writeUint64(uint64(token))
				writeUint64(uint64(transitions[token] + 1))
			}
		}
	}
	// Preserve v2 signature byte-for-byte when no GBNF is configured so
	// existing sampler/session states remain loadable
	if config.GBNF != nil {
		writeUint64(^uint64(0))
		writeUint64(config.GBNF.signature)
	}
	if config.TopNSigma > 0 || config.XTCProbability > 0 || config.MinKeep > 0 {
		writeUint64(^uint64(2))
		writeUint64(uint64(math.Float32bits(config.TopNSigma)))
		writeUint64(uint64(math.Float32bits(config.XTCProbability)))
		writeUint64(uint64(math.Float32bits(config.XTCThreshold)))
		writeUint64(uint64(config.MinKeep))
	}
	if config.DynatempRange > 0 {
		writeUint64(^uint64(3))
		writeUint64(uint64(math.Float32bits(config.DynatempRange)))
		writeUint64(uint64(math.Float32bits(config.DynatempExponent)))
	}
	for _, stage := range config.Samplers {
		if stage == SamplerAdaptiveP {
			writeUint64(^uint64(4))
			writeUint64(uint64(math.Float32bits(config.AdaptiveTarget)))
			writeUint64(uint64(math.Float32bits(config.AdaptiveDecay)))
			break
		}
	}
	if len(config.LogitBiases) > 0 {
		writeUint64(^uint64(5))
		writeUint64(uint64(len(config.LogitBiases)))
		for _, bias := range config.LogitBiases {
			writeUint64(uint64(bias.Token))
			writeUint64(uint64(math.Float32bits(bias.Bias)))
		}
	}
	if config.Infill != nil {
		writeUint64(^uint64(6))
		writeUint64(config.Infill.signature)
	}
	if !samplerOrdersEqual(config.Samplers, defaultSamplerOrder) {
		writeUint64(^uint64(1))
		writeUint64(uint64(len(config.Samplers)))
		for _, stage := range config.Samplers {
			writeUint64(uint64(len(stage)))
			_, _ = hash.Write([]byte(stage))
		}
	}
	return hash.Sum64()
}

func infillVocabularySignature(vocabulary *InfillVocabulary) uint64 {
	hash := fnv.New64a()
	var buffer [8]byte
	writeUint64 := func(value uint64) {
		binary.LittleEndian.PutUint64(buffer[:], value)
		_, _ = hash.Write(buffer[:])
	}
	writeUint64(uint64(len(vocabulary.Pieces)))
	for index, piece := range vocabulary.Pieces {
		writeUint64(uint64(len(piece)))
		_, _ = hash.Write([]byte(piece))
		if vocabulary.EOG[index] {
			writeUint64(1)
		} else {
			writeUint64(0)
		}
	}
	writeUint64(uint64(vocabulary.EOT + 1))
	writeUint64(uint64(vocabulary.EOS + 1))
	return hash.Sum64()
}

func samplerOrdersEqual(left, right []SamplerStage) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
