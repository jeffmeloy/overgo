//go:build windows

package trainingprogram_test

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// Objective-mixture protocol, written before the run (Probe Discipline): the
// incumbent objective trains on corpus A alone; the composed mixture
// interleaves corpora A and B at its 1:1 schedule. Both train scratch models
// under two seeds and are judged by held-out validation loss over the FULL
// corpus -- the descendants decide. The mixture, seeing both distributions,
// should beat the single-corpus incumbent on the combined held-out set; the
// promotion must be faithful to the measurement either way, and the schedule
// derivation must be deterministic and weight-proportional.
func TestWeightedObjectiveMixtureProducesBetterDescendant(t *testing.T) {
	cudatest.Require(t)
	objectiveA := testutil.ArtifactID(t, artifact.KindRecipe, "objective-corpus-a")
	objectiveB := testutil.ArtifactID(t, artifact.KindRecipe, "objective-corpus-b")
	mixture, err := trainingprogram.NewObjectiveMixture([]trainingprogram.MixtureComponent{
		{Objective: objectiveA, Weight: 1},
		{Objective: objectiveB, Weight: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	replay, err := trainingprogram.NewObjectiveMixture([]trainingprogram.MixtureComponent{
		{Objective: objectiveB, Weight: 1},
		{Objective: objectiveA, Weight: 1},
	})
	if err != nil || replay.ID != mixture.ID {
		t.Fatalf("mixture identity not order-canonical: (%v, %v)", replay.ID, err)
	}
	if _, err := trainingprogram.NewObjectiveMixture([]trainingprogram.MixtureComponent{
		{Objective: objectiveA, Weight: 1},
	}); err == nil {
		t.Fatal("single-component mixture accepted")
	}
	weighted, err := trainingprogram.NewObjectiveMixture([]trainingprogram.MixtureComponent{
		{Objective: objectiveA, Weight: 3},
		{Objective: objectiveB, Weight: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	schedule := weighted.Schedule()
	counts := map[int]int{}
	for _, index := range schedule {
		counts[index]++
	}
	low, high := counts[0], counts[1]
	if low > high {
		low, high = high, low
	}
	// Components canonicalize in ID order, so the 3-weight component may sit
	// at either index; the schedule must realize the weight multiset exactly.
	if len(schedule) != 4 || low != 1 || high != 3 {
		t.Fatalf("schedule %v does not realize the 3:1 weights", schedule)
	}

	corpusA := []string{
		"the quick brown fox jumps over the lazy dog",
		"pack my box with five dozen liquor jugs",
		"how vexingly quick daft zebras jump",
	}
	corpusB := []string{
		"sphinx of black quartz judge my vow",
		"the five boxing wizards jump quickly",
		"jackdaws love my big sphinx of quartz",
	}
	combined := append(append([]string(nil), corpusA...), corpusB...)
	const steps = 60
	started := time.Now()
	train := func(documents []string, mixed bool, seed int64) float64 {
		construction, err := scratchmodel.Compile(
			scratchmodel.CorpusFacts{Documents: combined, Seed: seed, Steps: steps},
			scratchmodel.AdaptiveDerivationProfile(),
		)
		if err != nil {
			t.Fatal(err)
		}
		trainer, err := scratchmodel.NewResidentTrainer(construction, steps)
		if err != nil {
			t.Fatal(err)
		}
		defer trainer.Close()
		schedule := mixture.Schedule()
		for step := 1; step <= steps; step++ {
			var document string
			if mixed {
				component := schedule[(step-1)%len(schedule)]
				pool := corpusA
				if component == 1 {
					pool = corpusB
				}
				document = pool[(step-1)%len(pool)]
			} else {
				document = documents[(step-1)%len(documents)]
			}
			tokens, err := construction.Tokens(document)
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
	for _, seed := range []int64{19, 23} {
		incumbentLoss := train(corpusA, false, seed)
		mixtureLoss := train(nil, true, seed)
		comparisons = append(comparisons, runrecord.SeedComparison{
			Seed: uint64(seed), Parent: incumbentLoss, Child: mixtureLoss,
		})
		t.Logf("seed %d: incumbent (corpus A only) held-out %.6f, mixture (A+B) held-out %.6f",
			seed, incumbentLoss, mixtureLoss)
	}
	contract := runrecord.MetricContract{
		Evaluator:   testutil.ArtifactID(t, artifact.KindEvidence, "mixture-heldout-suite"),
		Split:       testutil.ArtifactID(t, artifact.KindDatasetShard, "mixture-validation"),
		Comparisons: comparisons, CostNS: uint64(time.Since(started).Nanoseconds()),
		HoldoutQueries: uint64(len(comparisons) * 2), HoldoutBudget: 8,
	}
	promotion, err := runrecord.PromoteObjectiveMixture(mixture.ID, objectiveA, contract)
	if err != nil {
		t.Fatal(err)
	}
	contractHolds := runrecord.ValidateDescendantImprovement(contract) == nil
	if promotion.Promoted != contractHolds {
		t.Fatalf("promotion %v unfaithful to contract verdict %v: %s",
			promotion.Promoted, contractHolds, promotion.Reason)
	}
	content, err := promotion.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := runrecord.ParseMixturePromotion(content.Data)
	if err != nil || parsed.ID != promotion.ID || parsed.Promoted != promotion.Promoted {
		t.Fatalf("roundtrip = (%+v, %v)", parsed, err)
	}
	verdict := "REFUSED"
	if promotion.Promoted {
		verdict = "PROMOTED"
	}
	t.Logf("mixture promotion: %s -- %s (wall %s)", verdict, promotion.Reason, time.Since(started))
}
