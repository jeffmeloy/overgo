package plan

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

type completionFixture struct {
	t                 *testing.T
	repository        string
	store             *overgodb.Store
	parent            Plan
	preAdvance        Plan
	child             Plan
	item, step        string
	verify            string
	manifest          artifact.ID
	codeManifest      artifact.ID
	preparation       runrecord.GateLifecycle
	preparationCommit artifact.CommitID
	acceptance        string
	completionHash    string
}

func newCompletionFixture(t *testing.T, parent Plan, item, step string) *completionFixture {
	t.Helper()
	repository := t.TempDir()
	if err := os.Mkdir(filepath.Join(repository, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	runGit(t, repository, nil, "init", "-q")
	runGit(t, repository, nil, "config", "user.name", "completion-test")
	runGit(t, repository, nil, "config", "user.email", "completion@test.invalid")
	if err := Save(filepath.Join(repository, filepath.FromSlash(Path)), parent); err != nil {
		t.Fatal(err)
	}
	runGit(t, repository, nil, "add", "--", Path)
	runGit(t, repository, nil, "commit", "-q", "-m", "baseline plan")
	child := parent
	parentStep, found := exactPlanStep(parent, item, step)
	verify := "go test ./..."
	if found {
		var err error
		child, err = pruneCompletionFixture(parent, item, step)
		if err != nil {
			t.Fatal(err)
		}
		verify = parentStep.Verify
	}
	store, err := overgodb.Open(filepath.Join(repository, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "completion-test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(
		strings.Repeat("a", 64), environment.ID, time.Unix(1, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparationContent, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	preparationCommit, err := store.Commit(t.Context(), artifact.Batch{
		Key:      "completion/prepared/" + preparation.ID.String(),
		Contents: []artifact.Content{environmentContent, preparationContent},
		Lineage:  preparation.Lineage(),
	})
	if err != nil {
		t.Fatal(err)
	}
	return &completionFixture{
		t: t, repository: repository, store: store, parent: parent, preAdvance: parent, child: child,
		item: item, step: step, verify: verify, acceptance: verify,
		manifest:     testutil.ArtifactID(t, artifact.KindRecipe, "completion-manifest"),
		codeManifest: testutil.ArtifactID(t, artifact.KindProfile, "completion-code-manifest"),
		preparation:  preparation, preparationCommit: preparationCommit,
	}
}

func pruneCompletionFixture(document Plan, itemID, stepID string) (Plan, error) {
	document.Items = slices.Clone(document.Items)
	for itemIndex := range document.Items {
		if document.Items[itemIndex].ID != itemID {
			continue
		}
		document.Items[itemIndex].Steps = slices.Clone(document.Items[itemIndex].Steps)
		for stepIndex := range document.Items[itemIndex].Steps {
			if document.Items[itemIndex].Steps[stepIndex].ID != stepID {
				continue
			}
			document.Items[itemIndex].Steps = slices.Delete(document.Items[itemIndex].Steps, stepIndex, stepIndex+1)
			if len(document.Items[itemIndex].Steps) == 0 {
				document.Items = slices.Delete(document.Items, itemIndex, itemIndex+1)
			}
			return document, nil
		}
		return Plan{}, os.ErrNotExist
	}
	return Plan{}, os.ErrNotExist
}

func standardCompletionPlan() Plan {
	return Plan{Campaign: "completion", Doctrine: "fixture", Items: []Item{
		{ID: "root", Status: StatusOpen, Steps: []Step{{
			ID: "do", Status: StatusOpen, Verify: "go test ./...",
		}}},
		{ID: "dependent", Status: StatusOpen, Steps: []Step{{
			ID: "do", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"root/do"},
		}}},
	}}
}

func (fixture *completionFixture) commit(message []byte, publish bool) string {
	fixture.t.Helper()
	if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), fixture.child); err != nil {
		fixture.t.Fatal(err)
	}
	runGit(fixture.t, fixture.repository, nil, "add", "--", Path)
	runGit(fixture.t, fixture.repository, message, "commit", "-q", "--allow-empty", "-F", "-")
	fixture.completionHash = strings.TrimSpace(string(runGit(
		fixture.t, fixture.repository, nil, "rev-parse", "HEAD",
	)))
	if publish {
		publishCompletionAttempt(
			fixture.t, fixture.store, fixture.completionHash, fixture.item, fixture.step,
			fixture.verify, fixture.acceptance, fixture.manifest, fixture.codeManifest, fixture.preparation,
		)
	}
	return fixture.completionHash
}

func (fixture *completionFixture) canonicalMessage() []byte {
	fixture.t.Helper()
	manifestPlan := completionManifestPlan(fixture.t, fixture.codeManifest, fixture.preparation.TreeKey)
	fixture.manifest = manifestPlan.ID
	message, err := CompletionCommitMessage(
		[]byte("complete fixture"), fixture.preAdvance, fixture.item, fixture.step,
		fixture.manifest, fixture.codeManifest, fixture.preparation.ID, fixture.preparationCommit,
	)
	if err != nil {
		fixture.t.Fatal(err)
	}
	return message
}

func completionManifestPlan(t *testing.T, candidate artifact.ID, treeKey string) automationcheck.ManifestPlan {
	t.Helper()
	base, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("completion-base-manifest"))
	if err != nil {
		t.Fatal(err)
	}
	runner := func(context.Context, automationcheck.Invocation) (bool, string, error) {
		return false, "verified", nil
	}
	invocations, err := automationcheck.Plan([]automationcheck.Check{{
		Descriptor: automationcheck.Descriptor{Name: "acceptance", Phase: runrecord.PhaseTest, Always: true},
		Run:        runner,
	}}, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := automationcheck.BindManifestPlan(
		base, candidate, strings.Repeat("c", 64), treeKey,
		automationcheck.Surface{Identity: "completion-fixture"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}

func (fixture *completionFixture) legacyMessage() []byte {
	fixture.t.Helper()
	return []byte("complete fixture\n\n" +
		completionItemTrailer + ": " + fixture.item + "\n" +
		completionStepTrailer + ": " + fixture.step + "\n" +
		completionManifestTrailer + ": " + fixture.manifest.String() + "\n" +
		completionCodeManifestTrailer + ": " + fixture.codeManifest.String() + "\n" +
		completionVerifyTrailer + ": " + fixture.verify + "\n")
}

func (fixture *completionFixture) rotatePreparation(treeKey string, started time.Time) {
	fixture.t.Helper()
	preparation, err := runrecord.NewGatePreparation(treeKey, fixture.preparation.Environment, started)
	if err != nil {
		fixture.t.Fatal(err)
	}
	content, err := preparation.Content()
	if err != nil {
		fixture.t.Fatal(err)
	}
	preparationCommit, err := fixture.store.Commit(context.Background(), artifact.Batch{
		Key:      "completion/prepared/" + preparation.ID.String(),
		Contents: []artifact.Content{content},
		Lineage:  preparation.Lineage(),
	})
	if err != nil {
		fixture.t.Fatal(err)
	}
	fixture.preparation = preparation
	fixture.preparationCommit = preparationCommit
}

func publishCompletionAttempt(
	t *testing.T,
	store *overgodb.Store,
	commit, item, step string,
	verify, acceptanceVerify string,
	manifest, codeManifest artifact.ID,
	preparation runrecord.GateLifecycle,
) runrecord.AttemptRecord {
	return publishCompletionAttemptWithManifestAnalysis(
		t, store, commit, item, step, verify, acceptanceVerify,
		manifest, codeManifest, preparation, true,
	)
}

func publishCompletionAttemptWithManifestAnalysis(
	t *testing.T,
	store *overgodb.Store,
	commit, item, step string,
	verify, acceptanceVerify string,
	manifest, codeManifest artifact.ID,
	preparation runrecord.GateLifecycle,
	atomicAnalysis bool,
) runrecord.AttemptRecord {
	t.Helper()
	ctx := t.Context()
	environment := preparation.Environment
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "completion/dependencies/" + commit,
		Artifacts: []artifact.Descriptor{
			{ID: manifest}, {ID: codeManifest},
		},
	}); err != nil {
		t.Fatal(err)
	}
	acceptanceEvidence, err := runrecord.FormatCompletionAcceptanceEvidence(
		testevidence.VerifyPolicyV1, item+"/"+step, acceptanceVerify,
	)
	if err != nil {
		t.Fatal(err)
	}
	if verdict := strings.LastIndex(acceptanceEvidence, " verdict="); verdict >= 0 {
		acceptanceEvidence = acceptanceEvidence[:verdict] +
			" verdict=" + string(testevidence.VerdictBitwiseDeterministic)
	}
	gate, err := runrecord.NewGateRecord(
		manifest, environment, commit, runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{
			{Name: "acceptance", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1, Evidence: acceptanceEvidence},
			{Name: "commit", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: 1},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(
		preparation, commit, gate.Result.ID, runrecord.OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: item, PlanStep: step, Result: gate.Result.ID, Recipe: manifest,
		CodeCommit: commit, Outcome: runrecord.OutcomeSucceeded, WallNS: 1,
		CandidateManifest: codeManifest,
	})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := gate.Batch("completion/final/" + commit)
	if err != nil {
		t.Fatal(err)
	}
	attemptContent, err := attempt.Content()
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Contents = append(batch.Contents, finalizationContent, attemptContent)
	batch.Lineage = append(batch.Lineage, finalization.Lineage()...)
	batch.Lineage = append(batch.Lineage, attempt.Lineage()...)
	var delayedAnalysis *artifact.Batch
	manifestPlan := completionManifestPlan(t, codeManifest, preparation.TreeKey)
	if manifestPlan.ID == manifest {
		analysis, err := automationcheck.NewManifestAnalysis(
			codemanifest.Delta{Base: manifestPlan.BaseManifest, Candidate: manifestPlan.CandidateManifest},
			codemanifest.Impact{
				Base: manifestPlan.BaseManifest.String(), Candidate: manifestPlan.CandidateManifest.String(),
			},
			manifestPlan, automationcheck.SelectionMetrics{},
			automationcheck.MeasureManifest(1, 1, 0, 0, 1, 0, 0, 0),
		)
		if err != nil {
			t.Fatal(err)
		}
		analysisContent, err := analysis.Content()
		if err != nil {
			t.Fatal(err)
		}
		analysisLineage := artifact.Lineage{
			Child: gate.Result.ID, Parent: analysis.ID, Relation: artifact.RelationDependsOn,
		}
		if atomicAnalysis {
			batch.Contents = append(batch.Contents, analysisContent)
			batch.Lineage = append(batch.Lineage, analysisLineage)
		} else {
			delayedAnalysis = &artifact.Batch{
				Key:      "completion/analysis/" + commit,
				Contents: []artifact.Content{analysisContent},
				Lineage:  []artifact.Lineage{analysisLineage},
			}
		}
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if delayedAnalysis != nil {
		if _, err := store.Commit(ctx, *delayedAnalysis); err != nil {
			t.Fatal(err)
		}
	}
	return attempt
}

func resolveFixture(fixture *completionFixture, document Plan, revision string) (CompletionAuthority, error) {
	fixture.t.Helper()
	return ResolveCompletionAuthority(
		context.Background(), fixture.repository, revision, document, fixture.store,
	)
}

func TestPrunedDependencyRequiresGatedCompletion(t *testing.T) {
	t.Run("exact ancestor gate", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		authority, err := resolveFixture(fixture, fixture.child, "HEAD")
		if err != nil {
			t.Fatal(err)
		}
		if !authority.ProtectsRevision() {
			t.Fatal("prepared completion did not protect its exact resolved revision")
		}
		parentHash := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD^")))
		parentAuthority, err := resolveFixture(fixture, fixture.parent, parentHash)
		if err != nil {
			t.Fatal(err)
		}
		if parentAuthority.ProtectsRevision() || (CompletionAuthority{}).ProtectsRevision() {
			t.Fatal("pre-activation or zero authority reported a protected revision")
		}
		if item, step, open := Current(fixture.child, UnassignedRole, authority); !open || item.ID != "dependent" || step.ID != "do" {
			t.Fatalf("dispatch = %s/%s open=%v", item.ID, step.ID, open)
		}
		replayed := fixture.child
		replayed.Doctrine = "different plan"
		if _, _, open := Current(replayed, UnassignedRole, authority); open {
			t.Fatal("resolved completion authority replayed onto a different plan")
		}
	})
	t.Run("legacy ancestor gate", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.legacyMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("legacy completion after activation is relevance independent", func(t *testing.T) {
		initial := Plan{Campaign: "completion", Doctrine: "fixture", Items: []Item{
			{ID: "seed", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
			{ID: "legacy", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
			{ID: "dependent", Status: StatusOpen, Steps: []Step{{
				ID: "do", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"legacy/do"},
			}}},
		}}
		fixture := newCompletionFixture(t, initial, "seed", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		postSeed := fixture.child
		postLegacy, err := Advance(postSeed, "legacy", "do")
		if err != nil {
			t.Fatal(err)
		}
		fixture.preAdvance = postSeed
		fixture.child = postLegacy
		fixture.item, fixture.step = "legacy", "do"
		fixture.verify, fixture.acceptance = "go test ./...", "go test ./..."
		fixture.manifest = testutil.ArtifactID(t, artifact.KindRecipe, "post-seed-legacy-manifest")
		fixture.codeManifest = testutil.ArtifactID(t, artifact.KindProfile, "post-seed-legacy-code-manifest")
		fixture.rotatePreparation(strings.Repeat("c", 64), time.Unix(3, 0))
		fixture.commit(fixture.legacyMessage(), true)

		for name, document := range map[string]Plan{
			"relevant":   postLegacy,
			"irrelevant": {Campaign: "completion", Doctrine: "fixture"},
		} {
			t.Run(name, func(t *testing.T) {
				if _, err := resolveFixture(fixture, document, "HEAD"); err == nil ||
					!strings.Contains(err.Error(), "legacy trailers after prepared activation") {
					t.Fatalf("post-activation legacy policy error = %v", err)
				}
			})
		}
	})
	t.Run("added row with preparation", func(t *testing.T) {
		parent := Plan{Campaign: "transient", Doctrine: "fixture", Items: []Item{{
			ID: "dependent", Status: StatusOpen, Steps: []Step{{
				ID: "do", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"transient/do"},
			}},
		}}}
		fixture := newCompletionFixture(t, parent, "transient", "do")
		fixture.preAdvance.Items = append(slices.Clone(parent.Items), Item{
			ID: "transient", Status: StatusOpen,
			Steps: []Step{{ID: "do", Status: StatusOpen, Verify: fixture.verify}},
		})
		fixture.commit(fixture.canonicalMessage(), true)
		authority, err := resolveFixture(fixture, fixture.child, "HEAD")
		if err != nil {
			t.Fatal(err)
		}
		if item, _, open := Current(fixture.child, UnassignedRole, authority); !open || item.ID != "dependent" {
			t.Fatalf("added completion row did not release dependency: item=%s open=%v", item.ID, open)
		}
		reused := fixture.child
		reused.Items = append(slices.Clone(reused.Items), Item{
			ID: "transient", Status: StatusOpen,
			Steps: []Step{{ID: "again", Status: StatusOpen, Verify: "go test ./..."}},
		})
		if _, err := resolveFixture(fixture, reused, "HEAD"); err == nil || !strings.Contains(err.Error(), "cannot be reused") {
			t.Fatalf("completed added-item reuse error = %v", err)
		}
	})
	t.Run("setverify with completion", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.preAdvance.Items = slices.Clone(fixture.parent.Items)
		fixture.preAdvance.Items[0].Steps = slices.Clone(fixture.parent.Items[0].Steps)
		fixture.preAdvance.Items[0].Steps[0].Verify = "go test ./internal/plan"
		fixture.verify = fixture.preAdvance.Items[0].Steps[0].Verify
		fixture.acceptance = fixture.verify
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("rerank with completion", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.preAdvance.Items = []Item{fixture.parent.Items[1], fixture.parent.Items[0]}
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("missing parent legacy row", func(t *testing.T) {
		parent := Plan{Campaign: "transient", Doctrine: "fixture", Items: []Item{{
			ID: "dependent", Status: StatusOpen, Steps: []Step{{
				ID: "do", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"transient/do"},
			}},
		}}}
		fixture := newCompletionFixture(t, parent, "transient", "do")
		fixture.commit(fixture.legacyMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "lacks the exact completion row") {
			t.Fatalf("legacy missing-parent completion error = %v", err)
		}
	})
	t.Run("absence without attempt", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.canonicalMessage(), false)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "successful gate attempt") {
			t.Fatalf("missing attempt error = %v", err)
		}
	})
	t.Run("unknown identity", func(t *testing.T) {
		document := standardCompletionPlan()
		document.Items[1].Steps[0].DependsOn = []string{"ghost/do"}
		document.Items = document.Items[1:]
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		if _, err := resolveFixture(fixture, document, "HEAD"); err == nil || !strings.Contains(err.Error(), "ghost/do") {
			t.Fatalf("unknown dependency error = %v", err)
		}
	})
	t.Run("duplicate trailer", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		message := fixture.canonicalMessage()
		message = append(message, []byte(completionItemTrailer+": root\n")...)
		fixture.commit(message, true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "exactly once") {
			t.Fatalf("duplicate trailer error = %v", err)
		}
	})
	t.Run("mismatched attempt authority", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		message := fixture.canonicalMessage()
		fixture.manifest = testutil.ArtifactID(t, artifact.KindRecipe, "foreign-manifest")
		fixture.commit(message, true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "authorities differ") {
			t.Fatalf("mismatched attempt error = %v", err)
		}
	})
	t.Run("mutated preparation introduction commit", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		mutated := fixture.preparationCommit
		mutated[0] ^= 1
		message := replaceCompletionTrailer(
			fixture.canonicalMessage(), completionPreparationCommitTrailer, mutated.String(),
		)
		fixture.commit(message, true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil ||
			!strings.Contains(err.Error(), "differs from its introduction authority") {
			t.Fatalf("mutated preparation introduction commit error = %v", err)
		}
	})
	t.Run("foreign valid preparation introduction commit", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		foreign, err := fixture.store.Commit(t.Context(), artifact.Batch{
			Key: "completion/foreign-preparation-commit",
			Artifacts: []artifact.Descriptor{{
				ID: testutil.ArtifactID(t, artifact.KindEvidence, "foreign-preparation-commit"),
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		message := replaceCompletionTrailer(
			fixture.canonicalMessage(), completionPreparationCommitTrailer, foreign.String(),
		)
		fixture.commit(message, true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil ||
			!strings.Contains(err.Error(), "differs from its introduction authority") {
			t.Fatalf("foreign preparation introduction commit error = %v", err)
		}
	})
	t.Run("manifest analysis appended after final authority", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.canonicalMessage(), false)
		publishCompletionAttemptWithManifestAnalysis(
			t, fixture.store, fixture.completionHash, fixture.item, fixture.step,
			fixture.verify, fixture.acceptance, fixture.manifest, fixture.codeManifest,
			fixture.preparation, false,
		)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil ||
			!strings.Contains(err.Error(), "manifest analysis was not published atomically") {
			t.Fatalf("later manifest analysis error = %v", err)
		}
	})
	t.Run("forged verifier trailer", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		message := bytes.Replace(
			fixture.canonicalMessage(),
			[]byte(completionVerifyTrailer+": "+fixture.verify),
			[]byte(completionVerifyTrailer+": go test ./internal/plan"), 1,
		)
		fixture.commit(message, true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "snapshot differs") {
			t.Fatalf("forged verifier error = %v", err)
		}
	})
	t.Run("forged acceptance verifier", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.acceptance = "go test ./internal/plan"
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "acceptance evidence") {
			t.Fatalf("forged acceptance evidence error = %v", err)
		}
	})
	t.Run("forged acceptance verdict", func(t *testing.T) {
		parent := standardCompletionPlan()
		parent.Items[0].Steps[0].Verify = "go run ./cmd/device-lane"
		fixture := newCompletionFixture(t, parent, "root", "do")
		// The fixture publisher deliberately records the deterministic class;
		// the verifier itself derives the tolerance-bounded class.
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "acceptance evidence") {
			t.Fatalf("forged acceptance verdict error = %v", err)
		}
	})
	t.Run("unrelated row pruning", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.child.Items = []Item{{
			ID: "observer", Status: StatusOpen, Steps: []Step{{
				ID: "do", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"root/do"},
			}},
		}}
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "deleted a baseline") {
			t.Fatalf("multi-row pruning error = %v", err)
		}
	})
	t.Run("unrelated empty item pruning", func(t *testing.T) {
		parent := standardCompletionPlan()
		parent.Items = append(parent.Items, Item{ID: "empty", Status: StatusOpen})
		fixture := newCompletionFixture(t, parent, "root", "do")
		fixture.child.Items = slices.Delete(fixture.child.Items, len(fixture.child.Items)-1, len(fixture.child.Items))
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "deleted a baseline") {
			t.Fatalf("empty-item pruning error = %v", err)
		}
	})
	t.Run("unreferenced final row cannot hide empty item pruning", func(t *testing.T) {
		parent := Plan{Campaign: "completion", Doctrine: "fixture", Items: []Item{
			{ID: "root", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
			{ID: "empty", Status: StatusOpen},
		}}
		fixture := newCompletionFixture(t, parent, "root", "do")
		fixture.child.Items = nil
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "deleted a baseline") {
			t.Fatalf("unreferenced empty-item pruning error = %v", err)
		}
	})
	t.Run("unreferenced prepared format cannot omit preparation", func(t *testing.T) {
		parent := Plan{Campaign: "completion", Doctrine: "fixture", Items: []Item{{
			ID: "root", Status: StatusOpen,
			Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
		}}}
		fixture := newCompletionFixture(t, parent, "root", "do")
		message := removeCompletionTrailer(fixture.canonicalMessage(), completionPreparationTrailer)
		fixture.commit(message, true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "require gate preparation") {
			t.Fatalf("unreferenced malformed prepared format error = %v", err)
		}
	})
	t.Run("non ancestor", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		baseline := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, baseline); err == nil || !strings.Contains(err.Error(), "lacks gated ancestor") {
			t.Fatalf("side-branch completion error = %v", err)
		}
	})
	t.Run("cross repository evidence", func(t *testing.T) {
		source := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		source.commit(source.canonicalMessage(), true)
		other := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		if _, err := ResolveCompletionAuthority(
			t.Context(), other.repository, "HEAD", source.child, source.store,
		); err == nil || !strings.Contains(err.Error(), "lacks gated ancestor") {
			t.Fatalf("cross-repository completion evidence error = %v", err)
		}
	})
	t.Run("reserved and multiline writer input", func(t *testing.T) {
		manifest := testutil.ArtifactID(t, artifact.KindRecipe, "writer-manifest")
		codeManifest := testutil.ArtifactID(t, artifact.KindProfile, "writer-code-manifest")
		preparation := testutil.ArtifactID(t, artifact.KindEvidence, "writer-preparation")
		preparationCommit := artifact.CommitID{1}
		writerPlan := Plan{Items: []Item{{
			ID: "item", Status: StatusOpen,
			Steps: []Step{{ID: "step", Status: StatusOpen, Verify: "go test ./..."}},
		}}}
		if _, err := CompletionCommitMessage(
			[]byte("subject\n\nOvergo-Plan-Item: forged"), writerPlan, "item", "step",
			manifest, codeManifest, preparation, preparationCommit,
		); err == nil {
			t.Fatal("reserved operator trailer accepted")
		}
		writerPlan.Items[0].Steps[0].Verify = "go test ./...\nOvergo-Plan-Step: forged"
		if _, err := CompletionCommitMessage(
			[]byte("subject"), writerPlan, "item", "step",
			manifest, codeManifest, preparation, preparationCommit,
		); err == nil {
			t.Fatal("multiline verifier accepted")
		}
		writerPlan.Items[0].Steps[0].Verify = "go test ./..."
		if _, err := CompletionCommitMessage(
			[]byte("subject"), writerPlan, "item", "step",
			manifest, codeManifest, artifact.ID{}, preparationCommit,
		); err == nil {
			t.Fatal("missing preparation accepted")
		}
		if _, err := CompletionCommitMessage(
			[]byte("subject"), writerPlan, "item", "step",
			manifest, codeManifest, preparation, artifact.CommitID{},
		); err == nil {
			t.Fatal("missing preparation introduction commit accepted")
		}
	})
	t.Run("preparation parser", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		message := fixture.canonicalMessage()
		trailers, completion, err := parseCompletionTrailers(string(message))
		if err != nil || !completion || trailers.preparation != fixture.preparation.ID ||
			trailers.preparationCommit != fixture.preparationCommit {
			t.Fatalf(
				"new completion preparation = (%s, %s, %v, %v)",
				trailers.preparation, trailers.preparationCommit, completion, err,
			)
		}
		if trailers.item != fixture.item || trailers.step != fixture.step ||
			trailers.manifest != fixture.manifest || trailers.codeManifest != fixture.codeManifest ||
			trailers.preparation != fixture.preparation.ID ||
			trailers.preparationCommit != fixture.preparationCommit {
			t.Fatal("canonical completion tuple differs from the writer inputs")
		}
		if err := verifyCompletionSnapshot(fixture.preAdvance, trailers); err != nil {
			t.Fatalf("canonical completion snapshot refused: %v", err)
		}
		foreignCommit := fixture.preparationCommit
		foreignCommit[0] ^= 1
		if trailers.preparationCommit == foreignCommit {
			t.Fatal("canonical completion tuple accepted a foreign preparation commit")
		}
		badIndex := replaceCompletionTrailer(message, completionItemIndexTrailer, "1")
		badIndexTrailers, completion, err := parseCompletionTrailers(string(badIndex))
		if err != nil || !completion {
			t.Fatalf("parse mismatched completion item index: completion=%v err=%v", completion, err)
		}
		if err := verifyCompletionSnapshot(fixture.preAdvance, badIndexTrailers); err == nil {
			t.Fatal("mismatched completion item index accepted")
		}
		badSnapshot := replaceCompletionTrailer(message, completionSnapshotTrailer, "%%%")
		if _, _, err := parseCompletionTrailers(string(badSnapshot)); err == nil {
			t.Fatal("malformed completion item snapshot accepted")
		}
		altered := fixture.preAdvance
		altered.Items = slices.Clone(altered.Items)
		altered.Items[0].Title = "same-commit title edit"
		alteredMessage, err := CompletionCommitMessage(
			[]byte("complete fixture"), altered, fixture.item, fixture.step,
			fixture.manifest, fixture.codeManifest, fixture.preparation.ID, fixture.preparationCommit,
		)
		if err != nil {
			t.Fatal(err)
		}
		alteredTrailers, completion, err := parseCompletionTrailers(string(alteredMessage))
		if err != nil || !completion {
			t.Fatalf("parse altered completion snapshot: completion=%v err=%v", completion, err)
		}
		if err := verifyCompletionSnapshot(fixture.preAdvance, alteredTrailers); err == nil ||
			!strings.Contains(err.Error(), "snapshot differs") {
			t.Fatalf("mismatched completion item snapshot error = %v", err)
		}
		trailers, completion, err = parseCompletionTrailers(string(fixture.legacyMessage()))
		if err != nil || !completion || trailers.preparation.Valid() {
			t.Fatalf("legacy completion preparation = (%s, %v, %v)", trailers.preparation, completion, err)
		}
	})
	t.Run("malformed snapshot and mismatched index reject replay", func(t *testing.T) {
		for name, mutate := range map[string]func([]byte) []byte{
			"missing preparation": func(message []byte) []byte {
				return removeCompletionTrailer(message, completionPreparationTrailer)
			},
			"missing preparation commit": func(message []byte) []byte {
				return removeCompletionTrailer(message, completionPreparationCommitTrailer)
			},
			"invalid preparation commit": func(message []byte) []byte {
				return replaceCompletionTrailer(message, completionPreparationCommitTrailer, strings.Repeat("F", 64))
			},
			"snapshot": func(message []byte) []byte {
				return replaceCompletionTrailer(message, completionSnapshotTrailer, "%%%")
			},
			"index": func(message []byte) []byte {
				return replaceCompletionTrailer(message, completionItemIndexTrailer, "2")
			},
		} {
			t.Run(name, func(t *testing.T) {
				fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
				fixture.commit(mutate(fixture.canonicalMessage()), true)
				if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil {
					t.Fatal("malformed snapshot authority accepted")
				}
			})
		}
	})
	t.Run("revision option injection", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		if _, err := resolveFixture(fixture, fixture.parent, "--help"); err == nil || !strings.Contains(err.Error(), "resolve completion revision") {
			t.Fatalf("revision option injection error = %v", err)
		}
	})
	t.Run("shallow history", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		head := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
		if err := os.WriteFile(filepath.Join(fixture.repository, ".git", "shallow"), []byte(head+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := resolveFixture(fixture, fixture.parent, "HEAD"); err == nil || !strings.Contains(err.Error(), "shallow repository") {
			t.Fatalf("shallow history error = %v", err)
		}
	})
	t.Run("replacement object cannot hide completion identity", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		tree := strings.TrimSpace(string(runGit(
			t, fixture.repository, nil, "show", "-s", "--format=%T", fixture.completionHash,
		)))
		commitAndParents := strings.Fields(string(runGit(
			t, fixture.repository, nil, "rev-list", "--parents", "-n", "1", fixture.completionHash,
		)))
		arguments := []string{"commit-tree", tree}
		for _, parent := range commitAndParents[1:] {
			arguments = append(arguments, "-p", parent)
		}
		replacement := strings.TrimSpace(string(runGit(
			t, fixture.repository, []byte("masked completion\n"), arguments...,
		)))
		runGit(t, fixture.repository, nil, "replace", fixture.completionHash, replacement)

		reused := fixture.child
		reused.Items = append(reused.Items, Item{
			ID: "root", Status: StatusOpen,
			Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
		})
		if _, err := resolveFixture(fixture, reused, "HEAD"); err == nil || !strings.Contains(err.Error(), "replacement refs") {
			t.Fatalf("replacement authority was not refused: %v", err)
		}
	})
	t.Run("legacy grafts are refused", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		graftsPath := strings.TrimSpace(string(runGit(
			t, fixture.repository, nil, "rev-parse", "--path-format=absolute", "--git-path", "info/grafts",
		)))
		if err := os.MkdirAll(filepath.Dir(graftsPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(graftsPath, []byte("legacy graft state\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := resolveFixture(fixture, fixture.parent, "HEAD"); err == nil || !strings.Contains(err.Error(), "legacy Git grafts") {
			t.Fatalf("legacy graft history error = %v", err)
		}
	})
	t.Run("protected raw deletion cannot hide identity reuse", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		withVictim := fixture.child
		withVictim.Items = append(withVictim.Items, Item{
			ID: "victim", Status: StatusOpen,
			Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
		})
		if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), withVictim); err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, nil, "add", "--", Path)
		runGit(t, fixture.repository, nil, "commit", "-q", "-m", "add protected row")
		if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), fixture.child); err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, nil, "add", "--", Path)
		runGit(t, fixture.repository, nil, "commit", "-q", "-m", "raw row deletion")
		if _, err := resolveFixture(fixture, withVictim, "HEAD"); err == nil || !strings.Contains(err.Error(), "without gated completion") {
			t.Fatalf("raw deletion hid reused identity: %v", err)
		}
	})
}

func runGit(t *testing.T, repository string, input []byte, arguments ...string) []byte {
	t.Helper()
	command := exec.Command("git", arguments...)
	command.Dir = repository
	if input != nil {
		command.Stdin = bytes.NewReader(input)
	}
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(arguments, " "), err, strings.TrimSpace(string(output)))
	}
	return output
}

func replaceCompletionTrailer(message []byte, key, value string) []byte {
	lines := strings.Split(string(message), "\n")
	prefix := key + ": "
	for index, line := range lines {
		if strings.HasPrefix(line, prefix) {
			lines[index] = prefix + value
			break
		}
	}
	return []byte(strings.Join(lines, "\n"))
}

func removeCompletionTrailer(message []byte, key string) []byte {
	lines := strings.Split(string(message), "\n")
	prefix := key + ": "
	for index, line := range lines {
		if strings.HasPrefix(line, prefix) {
			lines = slices.Delete(lines, index, index+1)
			break
		}
	}
	return []byte(strings.Join(lines, "\n"))
}
