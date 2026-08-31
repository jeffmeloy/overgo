package runrecord

import (
	"context"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

type attemptGateChain struct {
	recipe       artifact.ID
	environment  artifact.ID
	preparation  GateLifecycle
	gate         GateRecord
	finalization GateLifecycle
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
	preparation, err := NewGatePreparation(strings.Repeat("a", 64), environmentID, time.Unix(1_000, 2))
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
		if _, err := artifact.CommitBatch(ctx, store, preparationBatch); err != nil {
			t.Fatal(err)
		}
	}

	gate, err := NewGateRecord(
		recipeID, environmentID, fixtureCodeCommit, OutcomeSucceeded, "", 10, steps,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := NewGateFinalization(
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
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyAttemptGateRequiresCanonicalLifecycle(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	chain := newAttemptGateChain(t, ctx, store, []GateStep{
		{Name: "acceptance", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 1},
		{Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1},
	})
	commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/final", chain, true, true, true, true)

	verified, err := VerifyAttemptGate(ctx, store, chain.attempt)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Gate.ID != chain.gate.Result.ID || verified.Run.ID != chain.gate.Run.ID ||
		verified.Preparation.ID != chain.preparation.ID || verified.Finalization.ID != chain.finalization.ID {
		t.Fatalf("verification = %+v", verified)
	}
	attempts, err := AttemptsForRecipe(ctx, store, chain.recipe)
	if err != nil || len(attempts) != 1 || attempts[0].ID != chain.attempt.ID {
		t.Fatalf("attempts for recipe = (%+v, %v)", attempts, err)
	}
	attempts, err = AttemptsForPreparation(ctx, store, chain.preparation.ID)
	if err != nil || len(attempts) != 1 || attempts[0].ID != chain.attempt.ID {
		t.Fatalf("attempts for preparation = (%+v, %v)", attempts, err)
	}
	finalization, found, err := GateFinalizationForPreparation(ctx, store, chain.preparation.ID)
	if err != nil || !found || finalization.ID != chain.finalization.ID {
		t.Fatalf("finalization = (%+v, %t, %v)", finalization, found, err)
	}
}

func TestVerifyAttemptGateRejectsIncompleteLifecycle(t *testing.T) {
	for _, test := range []struct {
		name                     string
		includeRun, includeFinal bool
	}{
		{name: "missing run", includeFinal: true},
		{name: "missing finalization", includeRun: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := t.Context()
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			chain := newAttemptGateChain(t, ctx, store, []GateStep{{
				Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1,
			}})
			commitAttemptGateChain(
				t, ctx, store, "fixture/attempt-gate/incomplete", chain,
				true, test.includeRun, test.includeFinal, true,
			)
			if _, err := VerifyAttemptGate(ctx, store, chain.attempt); err == nil {
				t.Fatal("incomplete gate lifecycle was accepted")
			}
		})
	}
}

func TestVerifyAttemptGateRejectsFailedFinalizationAndMissingCommit(t *testing.T) {
	t.Run("failed finalization", func(t *testing.T) {
		ctx := t.Context()
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		chain := newAttemptGateChain(t, ctx, store, []GateStep{{
			Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1,
		}})
		chain.finalization, err = NewGateFinalization(
			chain.preparation, fixtureCodeCommit, chain.gate.Result.ID, OutcomeFailed,
		)
		if err != nil {
			t.Fatal(err)
		}
		commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/failed-finalization", chain, true, true, true, true)
		if _, err := VerifyAttemptGate(ctx, store, chain.attempt); err == nil {
			t.Fatal("failed finalization was accepted")
		}
	})

	t.Run("mismatched preparation facts", func(t *testing.T) {
		ctx := t.Context()
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		chain := newAttemptGateChain(t, ctx, store, []GateStep{{
			Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1,
		}})
		preparationID, resultID := chain.preparation.ID, chain.gate.Result.ID
		chain.finalization, err = gateLifecycleCodec.New(GateLifecycle{
			Version: artifact.InitialDocumentVersion, State: GateFinalized,
			TreeKey: strings.Repeat("b", 64), Environment: chain.environment,
			Started:     time.Unix(2_000, 3).UTC().Format(time.RFC3339Nano),
			Preparation: &preparationID, CodeCommit: fixtureCodeCommit,
			Result: &resultID, Outcome: OutcomeSucceeded,
		})
		if err != nil {
			t.Fatal(err)
		}
		commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/mismatched-preparation", chain, true, true, true, true)
		if _, err := VerifyAttemptGate(ctx, store, chain.attempt); err == nil {
			t.Fatal("finalization with mismatched preparation facts was accepted")
		}
	})

	t.Run("missing commit step", func(t *testing.T) {
		ctx := t.Context()
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		chain := newAttemptGateChain(t, ctx, store, []GateStep{{
			Name: "acceptance", Phase: PhaseTest, Outcome: StepSucceeded, DurationNS: 1,
		}})
		commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/missing-commit", chain, true, true, true, true)
		if _, err := VerifyAttemptGate(ctx, store, chain.attempt); err == nil {
			t.Fatal("gate without the successful commit step was accepted")
		}
	})
}

func TestVerifyAttemptGateRejectsSplitIntroduction(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	chain := newAttemptGateChain(t, ctx, store, []GateStep{{
		Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1,
	}})
	commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/split/result", chain, true, true, false, false)
	commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/split/authority", chain, false, false, true, true)
	if _, err := VerifyAttemptGate(ctx, store, chain.attempt); err == nil {
		t.Fatal("split-batch attempt gate lifecycle was accepted")
	}
}

func TestVerifyAttemptGateRejectsPreparationPublishedWithFinalBatch(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	chain := newAttemptGateChain(t, ctx, store, []GateStep{{
		Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1,
	}}, false)
	preparationContent, err := chain.preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	gateContent, err := chain.gate.Result.Content()
	if err != nil {
		t.Fatal(err)
	}
	runContent, err := chain.gate.Run.Content()
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := chain.finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	attemptContent, err := chain.attempt.Content()
	if err != nil {
		t.Fatal(err)
	}
	lineage := append(chain.preparation.Lineage(), chain.gate.Result.Lineage()...)
	lineage = append(lineage, chain.gate.Run.Lineage()...)
	lineage = append(lineage, chain.finalization.Lineage()...)
	lineage = append(lineage, chain.attempt.Lineage()...)
	batch, err := artifact.NewDocumentBatch(
		"fixture/attempt-gate/post-hoc-preparation",
		[]artifact.Content{preparationContent, gateContent, runContent, finalizationContent, attemptContent},
		lineage,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyAttemptGate(ctx, store, chain.attempt); err == nil {
		t.Fatal("preparation published with the final attempt batch was accepted")
	}
}

func TestVerifyAttemptGateRejectsEnvironmentPublishedAfterPreparation(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "late-environment-recipe")
	testutil.PublishArtifact(t, store, recipeID)
	environment, err := NewEnvironment(Environment{
		Host: "late-environment", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/attempt-gate/late-environment/descriptor",
		Artifacts: []artifact.Descriptor{environmentContent.Descriptor},
	}); err != nil {
		t.Fatal(err)
	}
	preparation, err := NewGatePreparation(strings.Repeat("a", 64), environment.ID, time.Unix(1_000, 2))
	if err != nil {
		t.Fatal(err)
	}
	preparationContent, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	preparationBatch, err := artifact.NewDocumentBatch(
		"fixture/attempt-gate/late-environment/preparation",
		[]artifact.Content{preparationContent}, preparation.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, preparationBatch); err != nil {
		t.Fatal(err)
	}

	gate, err := NewGateRecord(
		recipeID, environment.ID, fixtureCodeCommit, OutcomeSucceeded, "", 10,
		[]GateStep{{Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1}},
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := NewGateFinalization(preparation, fixtureCodeCommit, gate.Result.ID, OutcomeSucceeded)
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
	chain := attemptGateChain{
		recipe: recipeID, environment: environment.ID, preparation: preparation,
		gate: gate, finalization: finalization, attempt: attempt,
	}
	commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/late-environment/final", chain, true, true, true, true)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/attempt-gate/late-environment/content", Contents: []artifact.Content{environmentContent},
	}); err != nil {
		t.Fatal(err)
	}

	if _, err := VerifyAttemptGate(ctx, store, attempt); err == nil ||
		!strings.Contains(err.Error(), "environment was published after its preparation") {
		t.Fatalf("late environment introduction error = %v", err)
	}
}

func TestVerifyAttemptGateRejectsAmbiguousBoundRun(t *testing.T) {
	for _, test := range []struct {
		name, codeCommit string
	}{
		{name: "second valid run", codeCommit: fixtureCodeCommit},
		{name: "contradictory typed run", codeCommit: strings.Repeat("b", 40)},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := t.Context()
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			chain := newAttemptGateChain(t, ctx, store, []GateStep{{
				Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1,
			}})
			extra, err := NewBoundRun(
				chain.recipe, OutcomeSucceeded, nil, []artifact.ID{chain.gate.Result.ID}, "", test.codeCommit,
				chain.environment, chain.gate.Run.MeasuredNS+1, chain.gate.Run.Phases,
			)
			if err != nil {
				t.Fatal(err)
			}
			commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/ambiguous-run", chain, true, true, true, true, extra)
			if _, err := VerifyAttemptGate(ctx, store, chain.attempt); err == nil {
				t.Fatal("ambiguous or contradictory bound gate runs were accepted")
			}
		})
	}
}

func TestVerifyAttemptGateRejectsConflictingPreparationFinalization(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	chain := newAttemptGateChain(t, ctx, store, []GateStep{{
		Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1,
	}})
	commitAttemptGateChain(t, ctx, store, "fixture/attempt-gate/final", chain, true, true, true, true)

	foreignResult := testutil.ArtifactID(t, artifact.KindEvidence, "conflicting-finalization-result")
	testutil.PublishArtifact(t, store, foreignResult)
	conflict, err := NewGateFinalization(
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
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := GateFinalizationForPreparation(ctx, store, chain.preparation.ID); err == nil {
		t.Fatal("conflicting finalizations were not rejected as ambiguous")
	}
	if _, err := AttemptsForPreparation(ctx, store, chain.preparation.ID); err == nil {
		t.Fatal("attempt lookup selected from conflicting finalization authority")
	}
	if _, err := VerifyAttemptGate(ctx, store, chain.attempt); err == nil {
		t.Fatal("preparation with a conflicting finalization was accepted")
	}
}

func TestCompletionAcceptanceEvidenceBindsExactVerify(t *testing.T) {
	verify := "go test ./internal/plan -run '^TestExact$' -count=1"
	reference := "authority-ratchet/complete.step"
	evidence, err := FormatCompletionAcceptanceEvidence(testevidence.VerifyPolicyV1, reference, verify)
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
		if _, err := FormatCompletionAcceptanceEvidence(
			testevidence.VerifyPolicyV1, invalid.reference, invalid.verify,
		); err == nil {
			t.Fatalf("invalid completion evidence authority was accepted: %+v", invalid)
		}
	}
	if _, err := FormatCompletionAcceptanceEvidence(
		testevidence.VerifyPolicy("verify-classifier-v2"), reference, verify,
	); err == nil {
		t.Fatal("unknown completion acceptance policy was accepted")
	}
}

func TestCompletionAcceptanceEvidenceReplaysFrozenPolicy(t *testing.T) {
	reference := "authority-ratchet/frozen-policy"
	verify := "go test ./internal/plan -run '^TestFutureDeviceMarker$' -count=1"
	evidence, err := FormatCompletionAcceptanceEvidence(testevidence.VerifyPolicyV1, reference, verify)
	if err != nil {
		t.Fatal(err)
	}
	simulatedNewerClassifier := func(string) testevidence.VerdictClass {
		return testevidence.VerdictToleranceBounded
	}
	v1, err := testevidence.ClassifyVerifyCommandForPolicy(testevidence.VerifyPolicyV1, verify)
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
		if _, err := FormatCompletionAcceptanceEvidence(testevidence.VerifyPolicyV1, reference, verify); err != nil {
			t.Fatalf("plan-compatible reference %q was rejected: %v", reference, err)
		}
	}
	for _, reference := range []string{
		"item", "/step", "item/", "item/step/extra", "item name/step", "item/step name",
		"item\\name/step", "item/step\\name", "item\tname/step",
	} {
		if _, err := FormatCompletionAcceptanceEvidence(testevidence.VerifyPolicyV1, reference, verify); err == nil {
			t.Fatalf("invalid plan reference %q was accepted", reference)
		}
	}
}
