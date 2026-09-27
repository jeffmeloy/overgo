package gate

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestAdmissionRunsBeforeTests holds the commit's completion audit to the
// acceptance step, before the test phases: a merge whose side history
// re-adds a completed row is refused by the check acceptance
// runs on the accepted candidate, the same refusal the commit would reach
// only after the full run.
func TestAdmissionRunsBeforeTests(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	publishSuccessfulInterruptedAttempt(t, fixture, true)
	if err := removeGateCommitIntent(fixture.repo); err != nil {
		t.Fatal(err)
	}
	mainBranch := recoveryGit(t, fixture.repo, "rev-parse", "--abbrev-ref", "HEAD")
	target := plan.Plan{Campaign: "interrupted-commit", Doctrine: "test recovery", Items: []plan.Item{{
		ID: "next", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/gate"}},
	}}}
	planPath := filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))
	commitPlan := func(document plan.Plan, message string) {
		t.Helper()
		if err := plan.Save(planPath, document); err != nil {
			t.Fatal(err)
		}
		runGitFixture(t, fixture.repo, "add", "--", plan.Path)
		runGitFixture(t, fixture.repo, "commit", "-q", "-m", message)
	}
	commitPlan(target, "add next row")
	runGitFixture(t, fixture.repo, "checkout", "-q", "-b", "raw-side-history")
	side := target
	side.Items = append(append([]plan.Item(nil), target.Items...), plan.Item{
		ID: "ratchet", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "recover", Status: plan.StatusOpen, Verify: "go test ./cmd/gate"}},
	})
	commitPlan(side, "raw re-add a completed row")
	sideParent := recoveryGit(t, fixture.repo, "rev-parse", "HEAD")
	runGitFixture(t, fixture.repo, "checkout", "-q", mainBranch)
	runGitFixture(t, fixture.repo, "commit", "-q", "--allow-empty", "-m", "clean main parent")
	mainParent := recoveryGit(t, fixture.repo, "rev-parse", "HEAD")
	store := openRecoveryStore(t, fixture)
	defer store.Close()
	g := &gateContext{
		repo: fixture.repo, planRef: "next/do", storePath: StorePath, environment: lifecycleTestEnvironment(t), store: store,
		authored:          []byte("merge\n\nCause: a fixture merge.\nPredicted effect: none.\n"),
		manifestPlan:      &automationcheck.ManifestPlan{ID: testutil.ArtifactID(t, artifact.KindRecipe, "early-admission-plan")},
		candidateManifest: &codemanifest.Manifest{ID: testutil.ArtifactID(t, artifact.KindProfile, "early-admission-code")},
		preparation:       runrecord.GateLifecycle{ID: testutil.ArtifactID(t, artifact.KindEvidence, "early-admission-preparation")},
		preparationCommit: artifact.CommitID{1},
		planHead:          mainParent, mergeBefore: &gateMergeIntent{Head: []byte(sideParent + "\n")},
		acceptedTree: recoveryGit(t, fixture.repo, "rev-parse", mainParent+"^{tree}"),
	}
	if err := g.admitCompletionEarly(); err == nil || !strings.Contains(err.Error(), "cannot be reused") {
		t.Fatalf("acceptance admitted a completion the commit refuses: %v", err)
	}
}
