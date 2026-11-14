//go:build windows

package scratchtrain

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/scratchmodel"
	"overgo/internal/scratchmodeltest"
	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

func TestProfilePromotedByDescendantQuality(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	candidate := scratchmodeltest.Profile(t)
	incumbent := candidate
	// The incumbent's normalization epsilon is degraded decisively: a
	// 2x nudge left the two 100-step losses inside floating-point
	// scheduling noise and the comparison flipped run to run. Promotion
	// is about quality differences that are real, so the fixture makes
	// one.
	incumbent.Epsilon = candidate.Epsilon * 1e6
	incumbent = testcheck.Must(scratchmodel.NewDerivationProfile(incumbent))(t)

	facts := scratchmodel.CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: 100}
	incumbentLoss := testcheck.Must(trainProfile(facts, incumbent))(t)
	candidateLoss := testcheck.Must(trainProfile(facts, candidate))(t)
	if candidateLoss >= incumbentLoss {
		t.Fatalf("candidate loss %.9f did not beat incumbent %.9f", candidateLoss, incumbentLoss)
	}
}

func trainProfile(facts scratchmodel.CorpusFacts, profile scratchmodel.DerivationProfile) (float64, error) {
	construction, err := scratchmodel.Compile(facts, profile)
	if err != nil {
		return 0, err
	}
	trainer, err := NewResidentTrainer(construction, facts.Steps, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		return 0, err
	}
	defer trainer.Close()
	train := construction.Split().Train
	for step := 1; step <= facts.Steps; step++ {
		tokens, tokenErr := construction.Tokens(train[(step-1)%len(train)])
		if tokenErr != nil {
			return 0, tokenErr
		}
		if _, stepErr := trainer.Step(tokens, step); stepErr != nil {
			return 0, stepErr
		}
	}
	validation := construction.Split().Validation
	loss := 0.0
	for _, document := range validation {
		tokens, tokenErr := construction.Tokens(document)
		if tokenErr != nil {
			return 0, tokenErr
		}
		value, evaluateErr := trainer.Evaluate(tokens)
		if evaluateErr != nil {
			return 0, evaluateErr
		}
		loss += value
	}
	return loss / float64(len(validation)), nil
}
