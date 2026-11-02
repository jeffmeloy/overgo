package trainingworkflow

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

// TestAPoolConfirmsAnObjectiveRun: a run that names a pool's acceptance trains, and its trained model answers on the
// pool's promotion part in place of a verdict. The answer is charged to the pool's budget, names the trained model, and
// reads a model that learned what the documents repeat as a gain, candidate less base; the run publishes no verdict. A
// pool fixed for another base is refused before anything trains, and so is a pool named by a proof run, which reads nothing.
func TestAPoolConfirmsAnObjectiveRun(t *testing.T) {
	t.Parallel()
	const (
		scored     = 4
		seenGain   = 2
		tokens     = 8
		steps      = 60
		parts      = 4
		wideWeight = 3
	)
	// Every prompt of two to six letters, each followed by what its last letter asks: a pool needs documents, all different.
	prompts := []string{"a", "b"}
	for start := 0; len(prompts[len(prompts)-1]) < 6; start++ {
		prompts = append(prompts, prompts[start]+"a", prompts[start]+"b")
	}
	var documents []string
	for _, prompt := range prompts[2:] {
		documents = append(documents, prompt+map[byte]string{'a': "cd", 'b': "dc"}[prompt[len(prompt)-1]])
	}
	text := registerScoredText(t, documents, trainingprogram.MetricTokenAccuracy, scored)
	ctx := t.Context()
	weights, err := IdentifyModelWeights(text.model)
	if err != nil {
		t.Fatal(err)
	}
	base, err := identifyAndRecordWeights(ctx, text.store, weights)
	if err != nil {
		t.Fatal(err)
	}
	derived := derivedIdentity(t, "overgo/pool-confirm-test/")
	over := func(objective trainingprogram.ObjectiveDocument, fixedFor artifact.ID) evaluation.HeldoutPoolRequest {
		return evaluation.HeldoutPoolRequest{
			Objective: objective, Base: fixedFor, Gain: seenGain, Sequence: tokens,
			Basis: []artifact.ID{derived(artifact.KindEvaluation, "a verdict")}, Weights: [parts]uint64{1, 1, wideWeight, wideWeight},
		}
	}
	// An objective whose verdicts read every held-out record has answered on all of them, so no pool starts over them.
	read := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	if _, err := evaluation.BuildHeldoutPool(ctx, read.store, over(read.objective, base)); err == nil || !strings.Contains(err.Error(), "no verdict has read") {
		t.Fatalf("a pool over records every verdict read = %v, want it refused", err)
	}
	publish := func(fixedFor artifact.ID) evaluation.HeldoutPool {
		pool, err := evaluation.BuildHeldoutPool(ctx, text.store, over(text.objective, fixedFor))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(ctx, text.store, pool.Batch); err != nil {
			t.Fatal(err)
		}
		return pool
	}
	output := filepath.Join(text.root, "trained")
	request := TextHeldoutRequest{Alias: text.alias, ModelDirectory: text.model, OutputDirectory: output, Steps: steps, MaximumSequence: tokens, Host: true}

	other := request
	other.Acceptance = publish(derived(artifact.KindModel, "another base")).Acceptance.ID
	if _, err := RunTextHeldout(ctx, text.store, other); err == nil || !strings.Contains(err.Error(), "was fixed for") {
		t.Fatalf("a pool fixed for another base = %v, want it refused", err)
	}
	if _, err := os.Stat(output); !os.IsNotExist(err) {
		t.Fatalf("a refused run wrote its output: %v", err)
	}

	pool := publish(base)
	acceptance := pool.Acceptance
	request.Acceptance = acceptance.ID
	proof := request
	proof.Proof = true
	if _, err := RunTextHeldout(ctx, text.store, proof); err == nil || !strings.Contains(err.Error(), "name no pool") {
		t.Fatalf("a proof run that names a pool = %v, want it refused", err)
	}
	result, err := RunTextHeldout(ctx, text.store, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Verdict.Valid() || result.Reads != 0 {
		t.Fatalf("a confirmed run published verdict %s and charged %d reads of the objective's records", result.Verdict, result.Reads)
	}
	confirmation := result.Confirmation
	if confirmation.Candidate != result.Trained || confirmation.Acceptance != acceptance.ID || confirmation.Verdict != "gain" {
		t.Fatalf("confirmation = %+v, want a gain for %s under %s", confirmation, result.Trained, acceptance.ID)
	}
	if result.ConfirmationsLeft != acceptance.Confirmations-1 {
		t.Fatalf("%d confirmations left of %d", result.ConfirmationsLeft, acceptance.Confirmations)
	}
	grant, err := runrecord.NewBudget("confirmation", pool.Partition.Promotion, acceptance.Confirmations, acceptance.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, charges, err := runrecord.BudgetCharges(ctx, text.store, grant.ID); err != nil || len(charges) != 1 {
		t.Fatalf("the pool was charged %d confirmations (%v), want the one", len(charges), err)
	}
}
