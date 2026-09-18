package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// Exercise execWorld against real Git transitions and the stored gate,
// attempt, preparation and finalization owners. No fake completion interface
// or production worktree/store supplies this integration test's authority.
func TestCampaignAcceptedProgress(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	git := func(args ...string) string {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", args...)
		command.Env = gitauthority.RepositoryEnvironment()
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	git("init", "-q")
	git("config", "user.name", "progress-test")
	git("config", "user.email", "progress@test.invalid")
	if err := os.MkdirAll("docs", 0o700); err != nil {
		t.Fatal(err)
	}
	const verify = "go test ./..."
	document := plan.Plan{Campaign: "progress fixture", Doctrine: "test", Items: []plan.Item{{ID: "grammar", Owner: "colibri", Status: plan.StatusOpen, Steps: []plan.Step{
		{ID: "do", Status: plan.StatusOpen, Verify: verify, DependsOn: []string{"grammar/binding"}},
		{ID: "binding", Status: plan.StatusOpen, Verify: verify},
	}}}}
	if err := plan.Save(plan.Path, document); err != nil {
		t.Fatal(err)
	}
	git("add", "--", plan.Path)
	git("commit", "-qm", "baseline")
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	reader, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	world := &execWorld{stopStore: reader}
	step := loop.Step{Item: "grammar", ID: "do"}
	before, err := world.ProgressCheckpoint()
	if err != nil {
		t.Fatal(err)
	}
	if accepted, err := world.AcceptedProgress(step, before); err != nil || accepted {
		t.Fatalf("empty history: %v %v", accepted, err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test"})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(strings.Repeat("a", 64), environment.ID, time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	envContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	prepContent, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "progress/prepared", Contents: []artifact.Content{envContent, prepContent}, Lineage: preparation.Lineage()}); err != nil {
		t.Fatal(err)
	}
	manifest := testutil.ArtifactID(t, artifact.KindRecipe, "progress manifest")
	candidate := testutil.ArtifactID(t, artifact.KindProfile, "progress code")
	document.Items[0].Steps = document.Items[0].Steps[:1]
	if err := plan.Save(plan.Path, document); err != nil {
		t.Fatal(err)
	}
	git("add", "--", plan.Path)
	// Legacy trailers also require the complete stored acceptance chain. The
	// plan owner's tests separately exercise the current prepared format.
	message := fmt.Sprintf("binding\n\nOvergo-Plan-Item: grammar\nOvergo-Plan-Step: binding\nOvergo-Manifest-Plan: %s\nOvergo-Code-Manifest: %s\nOvergo-Verify: %s\n", manifest, candidate, verify)
	git("commit", "-qm", message)
	completed := git("rev-parse", "HEAD")
	if accepted, err := world.AcceptedProgress(step, before); err == nil || accepted {
		t.Fatal("commit trailers without accepted store evidence earned progress")
	}
	evidence, err := runrecord.FormatCompletionAcceptanceEvidence(runrecord.VerifyPolicyV1, "grammar/binding", verify)
	if err != nil {
		t.Fatal(err)
	}
	gate, err := runrecord.NewGateRecord(manifest, environment.ID, completed, runrecord.OutcomeSucceeded, "", 1, []runrecord.GateStep{
		{Name: "acceptance", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1, Evidence: evidence},
		{Name: "commit", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(preparation, completed, gate.Result.ID, runrecord.OutcomeSucceeded)
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{PlanItem: "grammar", PlanStep: "binding", Result: gate.Result.ID, Recipe: manifest, CodeCommit: completed, Outcome: runrecord.OutcomeSucceeded, WallNS: 1, CandidateManifest: candidate})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := gate.Batch("progress/accepted")
	if err != nil {
		t.Fatal(err)
	}
	finalContent, err := finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	attemptContent, err := attempt.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: manifest}, artifact.Descriptor{ID: candidate})
	batch.Contents = append(batch.Contents, finalContent, attemptContent)
	batch.Lineage = append(batch.Lineage, finalization.Lineage()...)
	batch.Lineage = append(batch.Lineage, attempt.Lineage()...)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	assertProgress := func(want bool) {
		t.Helper()
		if accepted, err := world.AcceptedProgress(step, before); err != nil || accepted != want {
			t.Fatalf("accepted=%v want=%v err=%v", accepted, want, err)
		}
	}
	assertProgress(true)
	if accepted, err := world.AcceptedProgress(step, completed); err != nil || accepted {
		t.Fatalf("unchanged completion: %v %v", accepted, err)
	}
	pending, err := runrecord.NewGateLaneObligation(completed, preparation.ID, gate.Result.ID, []string{"test-device"}, []string{plan.Path}, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	var previous *artifact.ID
	for _, state := range []runrecord.LaneObligationState{runrecord.LaneObligationPending, runrecord.LaneObligationRunning, runrecord.LaneObligationFailed, runrecord.LaneObligationRunning, runrecord.LaneObligationPassed} {
		if state != runrecord.LaneObligationPending {
			outcome := artifact.ID{}
			if state == runrecord.LaneObligationFailed || state == runrecord.LaneObligationPassed {
				outcome = gate.Result.ID
			}
			pending, err = pending.Transition(state, outcome, time.Now())
			if err != nil {
				t.Fatal(err)
			}
		}
		batch, err := pending.Batch(previous)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
		id := pending.ID
		previous = &id
		assertProgress(state == runrecord.LaneObligationPassed)
	}
}
