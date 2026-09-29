package runrecord

import (
	"context"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

type attemptGateChain struct {
	recipe       artifact.ID
	environment  artifact.ID
	preparation  gateLifecycle
	gate         gateRecord
	finalization gateLifecycle
	attempt      AttemptRecord
}

func newAttemptGateChain(
	t testing.TB,
	ctx context.Context,
	store *overgodb.Store,
	steps []GateStep,
	publishPreparation ...bool,
) attemptGateChain {
	t.Helper()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "attempt-gate-recipe")
	testutil.PublishArtifact(t, store, recipeID)
	environmentID := publishGateTestEnvironment(t, ctx, store, "attempt-gate-environment")
	preparation, err := newGatePreparation(strings.Repeat("a", 64), environmentID, time.Unix(1_000, 2))
	if err != nil {
		t.Fatal(err)
	}
	preparationContent, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	if len(publishPreparation) == 0 || publishPreparation[0] {
		preparationBatch, err := artifact.NewDocumentBatch(
			"fixture/attempt-gate/preparation", []artifact.Content{preparationContent}, preparation.Lineage(), nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.CommitAs(ctx, gateDoor, preparationBatch); err != nil {
			t.Fatal(err)
		}
	}

	gate, err := NewGateRecord(
		recipeID, environmentID, fixtureCodeCommit, OutcomeSucceeded, "", 10, steps,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := newGateFinalization(
		preparation, fixtureCodeCommit, gate.Result.ID, OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := NewAttemptRecord(AttemptRecord{
		PlanItem: "authority", PlanStep: "complete", Result: gate.Result.ID, Recipe: recipeID,
		CodeCommit: fixtureCodeCommit, Outcome: OutcomeSucceeded, WallNS: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	return attemptGateChain{
		recipe: recipeID, environment: environmentID, preparation: preparation,
		gate: gate, finalization: finalization, attempt: attempt,
	}
}

func commitAttemptGateChain(
	t testing.TB,
	ctx context.Context,
	store *overgodb.Store,
	key string,
	chain attemptGateChain,
	gate, run, finalization, attempt bool,
	extraRuns ...Run,
) {
	t.Helper()
	var contents []artifact.Content
	var lineage []artifact.Lineage
	appendDocument := func(content artifact.Content, edges []artifact.Lineage, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, content)
		lineage = append(lineage, edges...)
	}
	if gate {
		content, err := chain.gate.Result.Content()
		appendDocument(content, chain.gate.Result.Lineage(), err)
	}
	if run {
		content, err := chain.gate.Run.Content()
		appendDocument(content, chain.gate.Run.Lineage(), err)
	}
	for _, extra := range extraRuns {
		content, err := extra.Content()
		appendDocument(content, extra.Lineage(), err)
	}
	if finalization {
		content, err := chain.finalization.Content()
		appendDocument(content, chain.finalization.Lineage(), err)
	}
	if attempt {
		content, err := chain.attempt.Content()
		appendDocument(content, chain.attempt.Lineage(), err)
	}
	batch, err := artifact.NewDocumentBatch(key, contents, lineage, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.CommitAs(ctx, gateDoor, batch); err != nil {
		t.Fatal(err)
	}
}

func TestGateFinalizationForPreparation(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	chain := newAttemptGateChain(t, ctx, store, []GateStep{
		{Name: "acceptance", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 1},
		{Name: "commit", Phase: phasePackage, Outcome: StepSucceeded, DurationNS: 1},
	})
	commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/final", chain, true, true, true, true)

	finalization, found, err := GateFinalizationForPreparation(ctx, store, chain.preparation.ID)
	if err != nil || !found || finalization.ID != chain.finalization.ID {
		t.Fatalf("finalization = (%+v, %t, %v)", finalization, found, err)
	}
}

func TestGateFinalizationForPreparationRejectsConflict(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	chain := newAttemptGateChain(t, ctx, store, []GateStep{{
		Name: "commit", Phase: phasePackage, Outcome: StepSucceeded, DurationNS: 1,
	}})
	commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/final", chain, true, true, true, true)

	foreignResult := testutil.ArtifactID(t, artifact.KindEvidence, "conflicting-finalization-result")
	testutil.PublishArtifact(t, store, foreignResult)
	conflict, err := newGateFinalization(
		chain.preparation, strings.Repeat("b", 40), foreignResult, OutcomeFailed,
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := conflict.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"fixture/attempt-gate/conflicting-finalization", []artifact.Content{content}, conflict.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.CommitAs(ctx, gateDoor, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := GateFinalizationForPreparation(ctx, store, chain.preparation.ID); err == nil {
		t.Fatal("conflicting finalizations were not rejected as ambiguous")
	}
}

func TestCompletionAcceptanceEvidenceBindsExactVerify(t *testing.T) {
	verify := "go test ./internal/plan -run '^TestExact$' -count=1"
	reference := "authority-ratchet/complete.step"
	evidence, err := formatCompletionAcceptanceEvidence(VerifyPolicyV1, reference, verify)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(evidence, "completion policy=verify-classifier-v1 plan="+reference+" verify_sha256=") ||
		!strings.HasSuffix(evidence, " verdict=bitwise-deterministic") {
		t.Fatalf("completion evidence = %q", evidence)
	}
	if err := VerifyCompletionAcceptanceEvidence(evidence, reference, verify); err != nil {
		t.Fatal(err)
	}
	for _, changed := range []struct{ reference, verify string }{
		{reference: "authority/other", verify: verify},
		{reference: reference, verify: verify + " -v"},
	} {
		if err := VerifyCompletionAcceptanceEvidence(evidence, changed.reference, changed.verify); err == nil {
			t.Fatalf("changed authority was accepted: %+v", changed)
		}
	}
	for _, mutated := range []string{
		strings.Replace(evidence, "verify-classifier-v1", "verify-classifier-v2", 1),
		strings.Replace(evidence, "verdict=bitwise-deterministic", "verdict=tolerance-bounded", 1),
		strings.TrimPrefix(evidence, "completion policy=verify-classifier-v1 "),
	} {
		if err := VerifyCompletionAcceptanceEvidence(mutated, reference, verify); err == nil {
			t.Fatalf("mutated completion acceptance evidence was accepted: %q", mutated)
		}
	}
	for _, invalid := range []struct{ reference, verify string }{
		{reference: "authority", verify: verify},
		{reference: reference, verify: verify + "\nextra"},
	} {
		if _, err := formatCompletionAcceptanceEvidence(
			VerifyPolicyV1, invalid.reference, invalid.verify,
		); err == nil {
			t.Fatalf("invalid completion evidence authority was accepted: %+v", invalid)
		}
	}
	if _, err := formatCompletionAcceptanceEvidence(
		VerifyPolicy("verify-classifier-v2"), reference, verify,
	); err == nil {
		t.Fatal("unknown completion acceptance policy was accepted")
	}
}

func TestCompletionAcceptanceEvidenceReplaysFrozenPolicy(t *testing.T) {
	reference := "authority-ratchet/frozen-policy"
	verify := "go test ./internal/plan -run '^TestFutureDeviceMarker$' -count=1"
	evidence, err := formatCompletionAcceptanceEvidence(VerifyPolicyV1, reference, verify)
	if err != nil {
		t.Fatal(err)
	}
	simulatedNewerClassifier := func(string) VerdictClass {
		return VerdictToleranceBounded
	}
	v1, err := ClassifyVerifyCommandForPolicy(VerifyPolicyV1, verify)
	if err != nil {
		t.Fatal(err)
	}
	if newer := simulatedNewerClassifier(verify); newer == v1 {
		t.Fatal("simulated newer classifier did not differ from frozen v1")
	}
	if err := VerifyCompletionAcceptanceEvidence(evidence, reference, verify); err != nil {
		t.Fatalf("historical v1 evidence changed under a simulated newer policy: %v", err)
	}
}

func TestCompletionAcceptanceEvidencePlanReferenceContract(t *testing.T) {
	verify := "go test ./internal/plan -count=1"
	for _, reference := range []string{
		"architecture-ratchets/completion.reference-v2",
		"RSI.v2/Guard-Phase:1",
	} {
		if _, err := formatCompletionAcceptanceEvidence(VerifyPolicyV1, reference, verify); err != nil {
			t.Fatalf("plan-compatible reference %q was rejected: %v", reference, err)
		}
	}
	for _, reference := range []string{
		"item", "/step", "item/", "item/step/extra", "item name/step", "item/step name",
		"item\\name/step", "item/step\\name", "item\tname/step",
	} {
		if _, err := formatCompletionAcceptanceEvidence(VerifyPolicyV1, reference, verify); err == nil {
			t.Fatalf("invalid plan reference %q was accepted", reference)
		}
	}
}
