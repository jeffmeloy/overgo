// Package sequencescore owns selected sequence likelihood aggregation.
package sequencescore

import (
	"errors"
	"math"
)

type Score struct {
	LogProbability float64 `json:"log_probability"`
	Tokens         uint64  `json:"tokens"`
	Characters     uint64  `json:"characters,omitzero"`
}

type Normalization string

const (
	NormalizationSum  Normalization = "sum"
	NormalizationMean Normalization = "mean"
	// NormalizationCharacters divides by the declared source character count.
	NormalizationCharacters Normalization = "characters"
)

type Selection struct {
	Index  int
	Values []float64
	Tied   bool
}

func (s Score) Valid() bool {
	return s.Tokens > 0 && !math.IsNaN(s.LogProbability) && !math.IsInf(s.LogProbability, 0)
}

type Accumulator struct {
	result Score
}

func (a *Accumulator) Observe(logProbability float64) error {
	if a == nil || math.IsNaN(logProbability) || math.IsInf(logProbability, 0) ||
		a.result.Tokens == math.MaxUint64 || math.IsNaN(a.result.LogProbability+logProbability) ||
		math.IsInf(a.result.LogProbability+logProbability, 0) {
		return errors.New("sequence score: invalid observation")
	}
	a.result.LogProbability += logProbability
	a.result.Tokens++
	return nil
}

func (a Accumulator) Result() (Score, error) {
	if !a.result.Valid() {
		return Score{}, errors.New("sequence score: observations are absent")
	}
	return a.result, nil
}

func Selected(logProbabilities []float64, selected []bool) (Score, error) {
	if len(logProbabilities) == 0 || len(logProbabilities) != len(selected) {
		return Score{}, errors.New("sequence score: selection extent differs")
	}
	var accumulator Accumulator
	for index, value := range logProbabilities {
		if selected[index] {
			if err := accumulator.Observe(value); err != nil {
				return Score{}, err
			}
		}
	}
	return accumulator.Result()
}

func Select(scores []Score, normalization Normalization) (Selection, error) {
	if len(scores) < 2 || normalization != NormalizationSum && normalization != NormalizationMean && normalization != NormalizationCharacters {
		return Selection{}, errors.New("sequence score: invalid choice contract")
	}
	result := Selection{Values: make([]float64, len(scores))}
	maximum := math.Inf(-1)
	for index, score := range scores {
		if !score.Valid() {
			return Selection{}, errors.New("sequence score: invalid choice score")
		}
		value := score.LogProbability
		if normalization == NormalizationMean {
			value /= float64(score.Tokens)
		} else if normalization == NormalizationCharacters {
			if score.Characters == 0 {
				return Selection{}, errors.New("sequence score: character denominator is absent")
			}
			value /= float64(score.Characters)
		}
		result.Values[index] = value
		if value > maximum {
			maximum, result.Index, result.Tied = value, index, false
		} else if value == maximum {
			result.Tied = true
		}
	}
	return result, nil
}
