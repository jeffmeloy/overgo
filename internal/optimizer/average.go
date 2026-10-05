package optimizer

import (
	"errors"
	"fmt"
	"slices"
)

// The warm-up of an average's decay: after n updates it is at most
// (averageWarmupStart+n)/(averageWarmupLag+n), the schedule TensorFlow's moving
// averages and ai-toolkit's EMA use, so an average of a few updates follows
// them rather than staying close to where it started.
const (
	averageWarmupStart = 1
	averageWarmupLag   = 10
)

// Average is an exponential moving average of a flat weight vector, taken
// once after each optimizer update: it starts at the weights it is given and
// after update n becomes d·average + (1-d)·weights, where d is the decay, held
// down early by the warm-up so it never exceeds (1+n)/(10+n).
type Average struct {
	decay   float64
	updates int
	values  []float32
}

// NewAverage starts an average of the weights at them, under a decay in (0, 1).
func NewAverage(weights []float32, decay float64) (*Average, error) {
	if !(decay > 0 && decay < 1) {
		return nil, fmt.Errorf("optimizer: average decay %v is not in (0, 1)", decay)
	}
	return &Average{decay: decay, values: slices.Clone(weights)}, nil
}

// Update folds the weights after one more optimizer update into the average.
func (average *Average) Update(weights []float32) error {
	if len(weights) != len(average.values) {
		return errors.New("optimizer: the weights an average is updated with differ in length from the ones it began with")
	}
	average.updates++
	steps := float64(average.updates)
	keep := float32(min(average.decay, (averageWarmupStart+steps)/(averageWarmupLag+steps)))
	for index, weight := range weights {
		average.values[index] = keep*average.values[index] + (1-keep)*weight
	}
	return nil
}

// Values are the average as it stands, the average's own storage.
func (average *Average) Values() []float32 { return average.values }

// Updates is how many updates the average has taken.
func (average *Average) Updates() int { return average.updates }

// RestoreAverage resumes an average that stood at the values after the given
// number of updates, under the decay it was taken with: its next update warms up
// from there as the one that followed those updates would have.
func RestoreAverage(values []float32, updates int, decay float64) (*Average, error) {
	average, err := NewAverage(values, decay)
	if err != nil {
		return nil, err
	}
	if updates < 0 {
		return nil, fmt.Errorf("optimizer: an average cannot have taken %d updates", updates)
	}
	average.updates = updates
	return average, nil
}

// Decay is the decay the average was started under, before its warm-up.
func (average *Average) Decay() float64 { return average.decay }
