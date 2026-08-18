// Package sequencescore owns selected sequence likelihood aggregation.
package sequencescore

import (
	"errors"
	"math"
)

type Score struct {
	LogProbability float64 `json:"log_probability"`
	Tokens         uint64  `json:"tokens"`
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
