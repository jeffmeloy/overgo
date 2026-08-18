//go:build windows

package adaptiveparity_test

import (
	"testing"
	"time"

	"overgo/internal/controllertrain"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/runrecord"
)

// Descendant-improvement protocol, written before the run (Probe Discipline):
// two independent scratch initializations of the workflow controller train on
// the immutable corpus; each seed's INITIAL model is the parent measurement
// and its FINAL model the child, evaluated by the identical held-out suite
// (same evaluator and split identities across every measurement). The full
// metric contract judges promotion: every seed's child beats its parent, the
// mean improvement exceeds the observed parent noise envelope, no child
// crosses the critical floor, the wall cost is reported, and the holdout
// spend fits its budget. The same contract must REFUSE inverted evidence.
const (
	descendantSteps         = 200
	descendantHoldoutBudget = 8
)

var descendantSeeds = []int64{41, 43}

// TestDescendantBeatsParent passes when the trained children satisfy the
// full metric contract and the contract refuses regression evidence.
func TestDescendantBeatsParent(t *testing.T) {
	cudatest.Require(t)
	commit := controllerSourceCommit(t)
	corpus, err := controllertrain.Compile(controllertrain.Spec{
		Train: controllerRecords(commit, false), Holdout: controllerRecords(commit, true),
	})
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	comparisons := make([]runrecord.SeedComparison, 0, len(descendantSeeds))
	var evaluator, split = corpus.Dataset(), corpus.Split()
	for _, seed := range descendantSeeds {
		run, err := controllertrain.TrainSeed(corpus, seed, descendantSteps)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if run.Evaluation.Recipe != run.InitialEvaluation.Recipe ||
			run.Evaluation.Dataset != run.InitialEvaluation.Dataset {
			t.Fatalf("seed %d parent and child were not measured under identical evaluator/split identities", seed)
		}
		evaluator, split = run.Evaluation.Recipe, corpus.Split()
		comparisons = append(comparisons, runrecord.SeedComparison{
			Seed: uint64(seed), Parent: run.Evidence.Initial.Loss, Child: run.Evidence.Final.Loss,
		})
		t.Logf("seed %d: parent loss %.6f -> child loss %.6f (action %.3f -> %.3f)",
			seed, run.Evidence.Initial.Loss, run.Evidence.Final.Loss,
			run.Evidence.Initial.ActionAccuracy, run.Evidence.Final.ActionAccuracy)
	}
	contract := runrecord.MetricContract{
		Evaluator: evaluator, Split: split,
		Comparisons:    comparisons,
		CriticalFloor:  comparisons[0].Parent, // no child may regress past its parent's starting loss
		CostNS:         uint64(time.Since(started).Nanoseconds()),
		HoldoutQueries: uint64(len(descendantSeeds) * 2),
		HoldoutBudget:  descendantHoldoutBudget,
	}
	if err := runrecord.ValidateDescendantImprovement(contract); err != nil {
		t.Fatalf("trained descendants failed the metric contract: %v", err)
	}

	inverted := contract
	inverted.Comparisons = make([]runrecord.SeedComparison, len(comparisons))
	for index, comparison := range comparisons {
		inverted.Comparisons[index] = runrecord.SeedComparison{
			Seed: comparison.Seed, Parent: comparison.Child, Child: comparison.Parent,
		}
	}
	if err := runrecord.ValidateDescendantImprovement(inverted); err == nil {
		t.Fatal("the contract promoted a regression")
	}
	overspent := contract
	overspent.HoldoutQueries = descendantHoldoutBudget + 1
	if err := runrecord.ValidateDescendantImprovement(overspent); err == nil {
		t.Fatal("the contract ignored a holdout budget overrun")
	}
	unreported := contract
	unreported.CostNS = 0
	if err := runrecord.ValidateDescendantImprovement(unreported); err == nil {
		t.Fatal("the contract accepted unreported resource cost")
	}
	t.Logf("metric contract satisfied: %d seeds, evaluator %s, split %s, wall %s",
		len(comparisons), evaluator, split, time.Since(started))
}
