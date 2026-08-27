package automationpolicy

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func commitStrategyAttempt(t *testing.T, store *overgodb.Store, strategy string, outcome runrecord.Outcome, label string) {
	t.Helper()
	record := runrecord.AttemptRecord{
		PlanItem: "slot-work", PlanStep: "do", Strategy: strategy,
		Result:     testutil.ArtifactID(t, artifact.KindEvidence, "result-"+strategy+label+string(outcome)),
		Recipe:     testutil.ArtifactID(t, artifact.KindRecipe, "gate"),
		CodeCommit: strings.Repeat("ab", 20), Outcome: outcome, WallNS: 5,
	}
	if outcome == runrecord.OutcomeFailed {
		record.Failure = "verify"
	}
	published, err := runrecord.NewAttemptRecord(record)
	if err != nil {
		t.Fatal(err)
	}
	content, err := published.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/attempt/" + published.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
}

func commitEvidence(t *testing.T, store *overgodb.Store, label string) artifact.ID {
	t.Helper()
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: artifact.JSONMediaType, Schema: "overgo/test-evidence/v1",
	}
	content, err := contract.ContentBytes([]byte(`{"experiment":"` + label + `"}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/evidence/" + content.Descriptor.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	return content.Descriptor.ID
}

// TestPolicyLifecyclePromotesOnEvidence walks the whole sanctioned
// path: declared needs nothing, experimental needs a recorded
// experiment, verified needs repeated measured wins, first activation
// governs the slot, a measured-better challenger displaces it with the
// incumbent retained, and rollback reactivates the incumbent.
func TestPolicyLifecyclePromotesOnEvidence(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()

	declared, err := Declare(ctx, store, "gate-worker", "baseline")
	if err != nil || declared.Stage != StageDeclared {
		t.Fatalf("declare = (%+v, %v)", declared, err)
	}
	if _, err := Declare(ctx, store, "gate-worker", "rival"); err == nil {
		t.Fatal("second candidate mid-lifecycle was accepted")
	}
	if _, err := Promote(ctx, store, "gate-worker", nil); err == nil {
		t.Fatal("experimental promotion without evidence was accepted")
	}
	experimentID := commitEvidence(t, store, "baseline-trial")
	experimental, err := Promote(ctx, store, "gate-worker", []artifact.ID{experimentID})
	if err != nil || experimental.Stage != StageExperimental {
		t.Fatalf("experimental = (%+v, %v)", experimental, err)
	}
	if _, err := Promote(ctx, store, "gate-worker", nil); err == nil {
		t.Fatal("verification without measured wins was accepted")
	}
	commitStrategyAttempt(t, store, "baseline", runrecord.OutcomeSucceeded, "")
	commitStrategyAttempt(t, store, "baseline", runrecord.OutcomeSucceeded, "x")
	commitStrategyAttempt(t, store, "baseline", runrecord.OutcomeFailed, "magics")
	verified, err := Promote(ctx, store, "gate-worker", nil)
	if err != nil || verified.Stage != StageVerified {
		t.Fatalf("verified = (%+v, %v)", verified, err)
	}
	active, err := Promote(ctx, store, "gate-worker", nil)
	if err != nil || active.Stage != StageActive || active.RollbackTo.Valid() {
		t.Fatalf("first activation = (%+v, %v)", active, err)
	}
	governing, standing, err := Load(ctx, store, "gate-worker")
	if err != nil || !standing || governing.ID != active.ID {
		t.Fatalf("governing = (%+v, %t, %v)", governing, standing, err)
	}

	// Challenger: better measured record displaces, retaining rollback.
	if _, err := Declare(ctx, store, "gate-worker", "challenger"); err != nil {
		t.Fatal(err)
	}
	challengerTrial := commitEvidence(t, store, "challenger-trial")
	if _, err := Promote(ctx, store, "gate-worker", []artifact.ID{challengerTrial}); err != nil {
		t.Fatal(err)
	}
	commitStrategyAttempt(t, store, "challenger", runrecord.OutcomeSucceeded, "")
	commitStrategyAttempt(t, store, "challenger", runrecord.OutcomeSucceeded, "y")
	commitStrategyAttempt(t, store, "challenger", runrecord.OutcomeSucceeded, "z")
	if _, err := Promote(ctx, store, "gate-worker", nil); err != nil {
		t.Fatal(err)
	}
	displacing, err := Promote(ctx, store, "gate-worker", nil)
	if err != nil || displacing.Stage != StageActive || displacing.RollbackTo != active.ID {
		t.Fatalf("displacement = (%+v, %v)", displacing, err)
	}

	restored, err := Rollback(ctx, store, "gate-worker")
	if err != nil || restored.ID != active.ID {
		t.Fatalf("rollback = (%+v, %v)", restored, err)
	}
	governing, _, err = Load(ctx, store, "gate-worker")
	if err != nil || governing.ID != active.ID {
		t.Fatalf("post-rollback governing = (%+v, %v)", governing, err)
	}
}

// TestPolicyRefusesUnmeasuredActivation pins the regression gate: a
// challenger measuring below the incumbent cannot activate, and a
// promotion citing evidence nobody recorded is refused.
func TestPolicyRefusesUnmeasuredActivation(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()

	if _, err := Promote(ctx, store, "empty-slot", nil); err == nil {
		t.Fatal("promotion of an undeclared slot was accepted")
	}
	if _, err := Declare(ctx, store, "slot", "strong"); err != nil {
		t.Fatal(err)
	}
	phantom := testutil.ArtifactID(t, artifact.KindEvidence, "never-recorded")
	if _, err := Promote(ctx, store, "slot", []artifact.ID{phantom}); err == nil {
		t.Fatal("promotion citing unrecorded evidence was accepted")
	}
	trial := commitEvidence(t, store, "strong-trial")
	for _, step := range []func() (Policy, error){
		func() (Policy, error) { return Promote(ctx, store, "slot", []artifact.ID{trial}) },
		func() (Policy, error) {
			commitStrategyAttempt(t, store, "strong", runrecord.OutcomeSucceeded, "")
			commitStrategyAttempt(t, store, "strong", runrecord.OutcomeSucceeded, "b")
			commitStrategyAttempt(t, store, "strong", runrecord.OutcomeSucceeded, "c")
			return Promote(ctx, store, "slot", nil)
		},
		func() (Policy, error) { return Promote(ctx, store, "slot", nil) },
	} {
		if _, err := step(); err != nil {
			t.Fatal(err)
		}
	}

	// Weak challenger: 1/3 success against the incumbent's 3/3.
	if _, err := Declare(ctx, store, "slot", "weak"); err != nil {
		t.Fatal(err)
	}
	weakTrial := commitEvidence(t, store, "weak-trial")
	if _, err := Promote(ctx, store, "slot", []artifact.ID{weakTrial}); err != nil {
		t.Fatal(err)
	}
	commitStrategyAttempt(t, store, "weak", runrecord.OutcomeSucceeded, "")
	commitStrategyAttempt(t, store, "weak", runrecord.OutcomeSucceeded, "d")
	commitStrategyAttempt(t, store, "weak", runrecord.OutcomeFailed, "test")
	if _, err := Promote(ctx, store, "slot", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Promote(ctx, store, "slot", nil); err == nil {
		t.Fatal("a challenger measuring below the incumbent activated")
	}
	if _, err := Rollback(ctx, store, "slot"); err == nil {
		t.Fatal("rollback without a retained incumbent was accepted")
	}
}
