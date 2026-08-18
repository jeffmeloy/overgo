//go:build windows

package scratchmodel

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// Derivation-profile promotion protocol, written before the run (Probe
// Discipline): the incumbent policy (AdaptiveDerivationProfile) and a
// candidate variant (doubled MLP budget) each derive and train scratch models
// from the identical corpus under two seeds; each profile is judged SOLELY on
// its models' held-out validation loss. The promotion document promotes the
// candidate only when its models beat the incumbent's on every seed with
// margin beyond the incumbent's observed noise -- the descendant metric
// contract -- and records a measured refusal otherwise. The test passes when
// the judgment is faithful to the measurements, whichever way they fall.
func TestDerivationProfilePromotedByDescendantQuality(t *testing.T) {
	cudatest.Require(t)
	incumbentProfile := AdaptiveDerivationProfile()
	candidateProfile := AdaptiveDerivationProfile()
	candidateProfile.Version = "adaptive-corpus-derivation-v1-wide-mlp"
	candidateProfile.MLPBudget *= 2

	incumbent, err := NewDerivationProfileDocument(incumbentProfile)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := NewDerivationProfileDocument(candidateProfile)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := NewDerivationProfileDocument(candidateProfile)
	if err != nil || replay.ID != candidate.ID {
		t.Fatalf("profile identity not deterministic: (%v, %v)", replay.ID, err)
	}
	invalid := incumbentProfile
	invalid.Epsilon = 0
	if _, err := NewDerivationProfileDocument(invalid); err == nil {
		t.Fatal("invalid derivation profile identified")
	}

	documents := []string{
		"the quick brown fox jumps over the lazy dog",
		"pack my box with five dozen liquor jugs",
		"how vexingly quick daft zebras jump",
		"sphinx of black quartz judge my vow",
		"the five boxing wizards jump quickly",
		"jackdaws love my big sphinx of quartz",
	}
	const steps = 60
	started := time.Now()
	heldOut := func(profile DerivationProfile, seed int64) float64 {
		construction, err := Compile(CorpusFacts{Documents: documents, Seed: seed, Steps: steps}, profile)
		if err != nil {
			t.Fatal(err)
		}
		trainer, err := NewResidentTrainer(construction, steps)
		if err != nil {
			t.Fatal(err)
		}
		defer trainer.Close()
		train := construction.Split().Train
		for step := 1; step <= steps; step++ {
			tokens, err := construction.Tokens(train[(step-1)%len(train)])
			if err != nil {
				t.Fatal(err)
			}
			if _, err := trainer.Step(tokens, step); err != nil {
				t.Fatal(err)
			}
		}
		total, count := 0.0, 0
		for _, document := range construction.Split().Validation {
			tokens, err := construction.Tokens(document)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := trainer.Evaluate(tokens)
			if err != nil {
				t.Fatal(err)
			}
			total += loss
			count++
		}
		if count == 0 {
			t.Fatal("no validation documents")
		}
		return total / float64(count)
	}
	comparisons := make([]runrecord.SeedComparison, 0, 2)
	for _, seed := range []int64{31, 37} {
		parent := heldOut(incumbentProfile, seed)
		child := heldOut(candidateProfile, seed)
		comparisons = append(comparisons, runrecord.SeedComparison{
			Seed: uint64(seed), Parent: parent, Child: child,
		})
		t.Logf("seed %d: incumbent held-out %.6f candidate held-out %.6f", seed, parent, child)
	}
	contract := runrecord.MetricContract{
		Evaluator:   testutil.ArtifactID(t, artifact.KindEvidence, "derivation-heldout-suite"),
		Split:       testutil.ArtifactID(t, artifact.KindDatasetShard, "derivation-validation"),
		Comparisons: comparisons, CostNS: uint64(time.Since(started).Nanoseconds()),
		HoldoutQueries: uint64(len(comparisons) * 2), HoldoutBudget: 8,
	}
	promotion, err := PromoteDerivationProfile(candidate, incumbent, contract)
	if err != nil {
		t.Fatal(err)
	}
	contractHolds := runrecord.ValidateDescendantImprovement(contract) == nil
	if promotion.Promoted != contractHolds {
		t.Fatalf("promotion %v is unfaithful to the contract verdict %v: %s",
			promotion.Promoted, contractHolds, promotion.Reason)
	}
	if promotion.Candidate != candidate.ID || promotion.Incumbent != incumbent.ID {
		t.Fatalf("promotion binding = %+v", promotion)
	}
	content, err := promotion.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseDerivationPromotion(content.Data)
	if err != nil || parsed.ID != promotion.ID || parsed.Promoted != promotion.Promoted {
		t.Fatalf("promotion roundtrip = (%+v, %v)", parsed, err)
	}
	if _, err := PromoteDerivationProfile(candidate, candidate, contract); err == nil {
		t.Fatal("self-challenge accepted")
	}
	verdict := "REFUSED"
	if promotion.Promoted {
		verdict = "PROMOTED"
	}
	t.Logf("derivation promotion: %s -- %s (wall %s)", verdict, promotion.Reason, time.Since(started))
}
