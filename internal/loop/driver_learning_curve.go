package loop

import (
	"errors"
	"fmt"
	"math"
)

// DriverAttemptMeasurement is one recorded composition-driver attempt with
// its measured outcome and cost: whether the fitness gate passed and the
// promotion landed, the measured fitness delta the promotion earned, the
// total compute the attempt spent, and the adapter-training share of it.
type DriverAttemptMeasurement struct {
	Fit            bool    `json:"fit"`
	Promoted       bool    `json:"promoted"`
	FitnessDelta   float64 `json:"fitness_delta"`
	ComputeNS      uint64  `json:"compute_ns"`
	AdapterTrainNS uint64  `json:"adapter_train_ns"`
}

// DriverLearningPoint is one cumulative point on the curve.
type DriverLearningPoint struct {
	Attempt             uint64  `json:"attempt"`
	Promotions          uint64  `json:"promotions"`
	HitRate             float64 `json:"hit_rate"`
	CumulativeFitness   float64 `json:"cumulative_fitness"`
	CumulativeComputeNS uint64  `json:"cumulative_compute_ns"`
}

// DriverLearningCurve is the go/no-go instrument for widening driver
// autonomy: candidate hit-rate, fitness delta per unit compute, and the
// adapter-training cost share over the recorded attempts, with the full
// cumulative curve and the early-versus-late hit-rate split the autonomy
// ratchet judges trend on.
type DriverLearningCurve struct {
	Attempts          uint64                `json:"attempts"`
	Promotions        uint64                `json:"promotions"`
	HitRate           float64               `json:"hit_rate"`
	FitnessPerCompute float64               `json:"fitness_per_compute"`
	AdapterTrainShare float64               `json:"adapter_train_share"`
	EarlyHitRate      float64               `json:"early_hit_rate"`
	LateHitRate       float64               `json:"late_hit_rate"`
	Points            []DriverLearningPoint `json:"points"`
}

// DeriveDriverLearningCurve aggregates recorded driver attempts in their
// recorded order. Every attempt must carry a measured compute cost — an
// unmeasured attempt cannot inform a go/no-go instrument and refuses —
// a promotion must have passed the fitness gate and carry a positive
// measured delta, and a refused attempt carries none. The early and late
// hit rates split the record in half so the ratchet can judge whether the
// driver is getting better without any distributional assumption.
func DeriveDriverLearningCurve(measurements []DriverAttemptMeasurement) (DriverLearningCurve, error) {
	if len(measurements) == 0 {
		return DriverLearningCurve{}, errors.New("loop: a learning curve requires recorded attempts")
	}
	curve := DriverLearningCurve{Attempts: uint64(len(measurements))}
	fitness := 0.0
	var compute, adapter uint64
	for index, measurement := range measurements {
		if measurement.ComputeNS == 0 || measurement.AdapterTrainNS > measurement.ComputeNS {
			return DriverLearningCurve{}, fmt.Errorf(
				"loop: attempt %d lacks a coherent measured compute cost; an unmeasured attempt cannot inform the instrument", index,
			)
		}
		if math.IsNaN(measurement.FitnessDelta) || math.IsInf(measurement.FitnessDelta, 0) {
			return DriverLearningCurve{}, fmt.Errorf("loop: attempt %d fitness delta is not finite", index)
		}
		if measurement.Promoted && (!measurement.Fit || measurement.FitnessDelta <= 0) {
			return DriverLearningCurve{}, fmt.Errorf(
				"loop: attempt %d claims a promotion without a fit verdict and a positive measured delta", index,
			)
		}
		if !measurement.Promoted && measurement.FitnessDelta != 0 {
			return DriverLearningCurve{}, fmt.Errorf(
				"loop: attempt %d carries a fitness delta without a promotion", index,
			)
		}
		if measurement.Promoted {
			curve.Promotions++
			fitness += measurement.FitnessDelta
		}
		compute += measurement.ComputeNS
		adapter += measurement.AdapterTrainNS
		curve.Points = append(curve.Points, DriverLearningPoint{
			Attempt:             uint64(index + 1),
			Promotions:          curve.Promotions,
			HitRate:             float64(curve.Promotions) / float64(index+1),
			CumulativeFitness:   fitness,
			CumulativeComputeNS: compute,
		})
	}
	curve.HitRate = float64(curve.Promotions) / float64(curve.Attempts)
	curve.FitnessPerCompute = fitness / float64(compute)
	curve.AdapterTrainShare = float64(adapter) / float64(compute)
	half := len(measurements) / 2
	if half > 0 {
		early, late := uint64(0), uint64(0)
		for index, measurement := range measurements {
			if !measurement.Promoted {
				continue
			}
			if index < half {
				early++
			} else {
				late++
			}
		}
		curve.EarlyHitRate = float64(early) / float64(half)
		curve.LateHitRate = float64(late) / float64(len(measurements)-half)
	}
	return curve, nil
}
