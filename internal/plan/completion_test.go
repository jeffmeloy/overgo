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
	"overgo/internal/testutil"
	"overgo/internal/worklease"
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
	message, err := CompletionCommitMessageWithMergeAuthority(
		[]byte("complete fixture"), fixture.preAdvance, fixture.item, fixture.step,
		fixture.manifest, fixture.codeManifest, fixture.preparation.ID, fixture.preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
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
	mergeAuthorities ...FirstParentTargetMergeAuthority,
) runrecord.AttemptRecord {
	return publishCompletionAttemptWithManifestAnalysis(
		t, store, commit, item, step, verify, acceptanceVerify,
		manifest, codeManifest, preparation, true, mergeAuthorities...,
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
	mergeAuthorities ...FirstParentTargetMergeAuthority,
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
		runrecord.VerifyPolicyV1, item+"/"+step, acceptanceVerify,
	)
	if err != nil {
		t.Fatal(err)
	}
	if verdict := strings.LastIndex(acceptanceEvidence, " verdict="); verdict >= 0 {
		acceptanceEvidence = acceptanceEvidence[:verdict] +
			" verdict=" + string(runrecord.VerdictBitwiseDeterministic)
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
	if len(mergeAuthorities) > 1 {
		t.Fatal("completion fixture accepts at most one projected merge authority")
	}
	if len(mergeAuthorities) == 1 {
		authority := mergeAuthorities[0]
		content, err := authority.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, authority.Lineage()...)
		batch.Lineage = append(batch.Lineage, artifact.Lineage{
			Child: gate.Result.ID, Parent: authority.ID, Relation: artifact.RelationDependsOn,
		})
	}
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
		if item, step, open := Current(fixture.child, worklease.UnassignedRole, authority); !open || item.ID != "dependent" || step.ID != "do" {
			t.Fatalf("dispatch = %s/%s open=%v", item.ID, step.ID, open)
		}
		replayed := fixture.child
		replayed.Doctrine = "different plan"
		if _, _, open := Current(replayed, worklease.UnassignedRole, authority); open {
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
		if item, _, open := Current(fixture.child, worklease.UnassignedRole, authority); !open || item.ID != "dependent" {
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
		if _, err := CompletionCommitMessageWithMergeAuthority(
			[]byte("subject\n\nOvergo-Plan-Item: forged"), writerPlan, "item", "step",
			manifest, codeManifest, preparation, preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
		); err == nil {
			t.Fatal("reserved operator trailer accepted")
		}
		writerPlan.Items[0].Steps[0].Verify = "go test ./...\nOvergo-Plan-Step: forged"
		if _, err := CompletionCommitMessageWithMergeAuthority(
			[]byte("subject"), writerPlan, "item", "step",
			manifest, codeManifest, preparation, preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
		); err == nil {
			t.Fatal("multiline verifier accepted")
		}
		writerPlan.Items[0].Steps[0].Verify = "go test ./..."
		if _, err := CompletionCommitMessageWithMergeAuthority(
			[]byte("subject"), writerPlan, "item", "step",
			manifest, codeManifest, artifact.ID{}, preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
		); err == nil {
			t.Fatal("missing preparation accepted")
		}
		if _, err := CompletionCommitMessageWithMergeAuthority(
			[]byte("subject"), writerPlan, "item", "step",
			manifest, codeManifest, preparation, artifact.CommitID{}, MergeProjectionSemanticUnion, artifact.ID{},
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
		if err := VerifyCompletionCommitMessageWithMergeAuthority(
			string(message), fixture.preAdvance, fixture.item, fixture.step,
			fixture.manifest, fixture.codeManifest, fixture.preparation.ID, fixture.preparationCommit,
			MergeProjectionSemanticUnion, artifact.ID{},
		); err != nil {
			t.Fatalf("canonical completion tuple refused: %v", err)
		}
		foreignCommit := fixture.preparationCommit
		foreignCommit[0] ^= 1
		if err := VerifyCompletionCommitMessageWithMergeAuthority(
			string(message), fixture.preAdvance, fixture.item, fixture.step,
			fixture.manifest, fixture.codeManifest, fixture.preparation.ID, foreignCommit,
			MergeProjectionSemanticUnion, artifact.ID{},
		); err == nil || !strings.Contains(err.Error(), "authority tuple") {
			t.Fatalf("foreign expected preparation commit error = %v", err)
		}
		badIndex := replaceCompletionTrailer(message, completionItemIndexTrailer, "1")
		if err := VerifyCompletionCommitMessageWithMergeAuthority(
			string(badIndex), fixture.preAdvance, fixture.item, fixture.step,
			fixture.manifest, fixture.codeManifest, fixture.preparation.ID, fixture.preparationCommit,
			MergeProjectionSemanticUnion, artifact.ID{},
		); err == nil {
			t.Fatal("mismatched completion item index accepted")
		}
		badSnapshot := replaceCompletionTrailer(message, completionSnapshotTrailer, "%%%")
		if _, _, err := parseCompletionTrailers(string(badSnapshot)); err == nil {
			t.Fatal("malformed completion item snapshot accepted")
		}
		altered := fixture.preAdvance
		altered.Items = slices.Clone(altered.Items)
		altered.Items[0].Title = "same-commit title edit"
		alteredMessage, err := CompletionCommitMessageWithMergeAuthority(
			[]byte("complete fixture"), altered, fixture.item, fixture.step,
			fixture.manifest, fixture.codeManifest, fixture.preparation.ID, fixture.preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
		)
		if err != nil {
			t.Fatal(err)
		}
		if err := VerifyCompletionCommitMessageWithMergeAuthority(
			string(alteredMessage), fixture.preAdvance, fixture.item, fixture.step,
			fixture.manifest, fixture.codeManifest, fixture.preparation.ID, fixture.preparationCommit,
			MergeProjectionSemanticUnion, artifact.ID{},
		); err == nil || !strings.Contains(err.Error(), "snapshot differs") {
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

func TestProtectedCompletionAuthorityBindsRetainedRevisionIdentities(t *testing.T) {
	fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
	fixture.commit(fixture.canonicalMessage(), true)

	withoutItem := fixture.child
	withoutItem.Items = nil
	withoutStep := fixture.child
	withoutStep.Items = slices.Clone(fixture.child.Items)
	withoutStep.Items[0].Steps = nil
	for name, document := range map[string]Plan{
		"item": withoutItem,
		"step": withoutStep,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := resolveFixture(fixture, document, "HEAD"); err == nil ||
				!strings.Contains(err.Error(), "live plan deleted an item or step retained at protected revision") {
				t.Fatalf("protected live-plan deletion error = %v", err)
			}
		})
	}
}

func TestLanePlanScope(t *testing.T) {
	baseline := standardCompletionPlan()
	baseline.Lane = "local"
	for i := range baseline.Items {
		baseline.Items[i].Owner = baseline.Lane
	}
	foreign := baseline.Items[0]
	foreign.ID, foreign.Owner = "foreign", "other"
	baseline.Items = append(baseline.Items, foreign)
	projected := baseline
	projected.Scope = ScopeLane
	projected.Items = slices.Clone(baseline.Items[:2])
	if err := Validate(projected); err != nil {
		t.Fatal(err)
	}
	if !preservesPlanIdentities(baseline, projected, true) || preservesPlanIdentities(baseline, projected, false) {
		t.Fatal("lane projection must preserve local identities without permitting raw deletion")
	}
	for name, mutate := range map[string]func(*Plan, *Plan){
		"implicit scope":  func(_, c *Plan) { c.Scope = "" },
		"changed lane":    func(_, c *Plan) { c.Lane = "other" },
		"unowned work":    func(b, _ *Plan) { b.Items[2].Owner = "" },
		"blank owner":     func(b, _ *Plan) { b.Items[2].Owner = " " },
		"unassigned work": func(b, _ *Plan) { b.Items[2].Owner = worklease.UnassignedRole },
		"local item":      func(_, c *Plan) { c.Items = c.Items[1:] },
		"local step":      func(_, c *Plan) { c.Items[0].Steps = nil },
		"partial foreign item": func(_, c *Plan) {
			retained := foreign
			retained.Owner, retained.Steps = c.Lane, nil
			c.Items = append(c.Items, retained)
		},
	} {
		t.Run(name, func(t *testing.T) {
			b, c := baseline, projected
			b.Items, c.Items = slices.Clone(b.Items), slices.Clone(c.Items)
			mutate(&b, &c)
			if preservesPlanIdentities(b, c, true) {
				t.Fatal("unsafe omission accepted")
			}
		})
	}
	t.Run("strict load", func(t *testing.T) {
		for _, invalid := range []Plan{{Scope: "unknown", Lane: "local"}, {Scope: ScopeLane}, {Scope: ScopeLane, Lane: worklease.UnassignedRole}, {Scope: ScopeLane, Lane: "local", Items: []Item{foreign}}} {
			if err := Validate(invalid); err == nil {
				t.Fatalf("invalid scope accepted: %+v", invalid)
			}
		}
		if _, err := MergeDocuments(projected, projected, baseline); err == nil {
			t.Fatal("merge imported a foreign row into lane scope")
		}
	})
	t.Run("protected live projection", func(t *testing.T) {
		fixture := newCompletionFixture(t, baseline, "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		live := fixture.child
		live.Scope = ScopeLane
		live.Items = slices.Clone(live.Items[:1])
		authority, err := resolveFixture(fixture, live, "HEAD")
		if err != nil || authority.completed("foreign/do") {
			t.Fatalf("live projection authority: %v", err)
		}
	})
	t.Run("gated projection and replay", func(t *testing.T) {
		fixture := newCompletionFixture(t, baseline, "root", "do")
		fixture.preAdvance = projected
		var err error
		fixture.child, err = Advance(projected, "root", "do")
		if err != nil {
			t.Fatal(err)
		}
		fixture.commit(fixture.canonicalMessage(), true)
		authority, err := resolveFixture(fixture, fixture.child, "HEAD")
		if err != nil {
			t.Fatal(err)
		}
		if !authority.completed("root/do") || authority.completed("foreign/do") {
			t.Fatal("projection changed completion credit")
		}
		dependent := fixture.child
		dependent.Items = slices.Clone(dependent.Items)
		dependent.Items[0].Steps = slices.Clone(dependent.Items[0].Steps)
		dependent.Items[0].Steps[0].DependsOn = []string{"foreign/do"}
		if _, err := resolveFixture(fixture, dependent, "HEAD"); err == nil {
			t.Fatal("omitted foreign item satisfied a prerequisite")
		}
	})
	t.Run("raw projection refused", func(t *testing.T) {
		fixture := newCompletionFixture(t, baseline, "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		fixture.child.Scope = ScopeLane
		fixture.child.Items = slices.Clone(fixture.child.Items[:1])
		fixture.commit([]byte("raw projection"), false)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "without gated completion") {
			t.Fatalf("raw projection error = %v", err)
		}
	})
}

func TestCompletionAuthorityRequiresExplicitCanonicalRepository(t *testing.T) {
	t.Run("ambient Git repository redirect", func(t *testing.T) {
		requested := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		redirect := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		t.Setenv("GIT_DIR", filepath.Join(redirect.repository, ".git"))
		t.Setenv("GIT_WORK_TREE", redirect.repository)

		authority, err := resolveFixture(requested, requested.parent, "HEAD")
		if err != nil {
			t.Fatal(err)
		}
		requestedInfo, err := os.Stat(requested.repository)
		if err != nil {
			t.Fatal(err)
		}
		resolvedInfo, err := os.Stat(authority.repository)
		if err != nil {
			t.Fatal(err)
		}
		if !os.SameFile(requestedInfo, resolvedInfo) {
			t.Fatalf("authority repository = %q, want explicit %q", authority.repository, requested.repository)
		}
	})

	t.Run("subdirectory", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		_, err := ResolveCompletionAuthority(
			t.Context(), filepath.Join(fixture.repository, "docs"), "HEAD", fixture.parent, fixture.store,
		)
		if err == nil || !strings.Contains(err.Error(), "exact repository root") {
			t.Fatalf("non-root completion repository error = %v", err)
		}
	})
}

func TestGitCompletionMessagesRejectRawCommitNUL(t *testing.T) {
	fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
	raw := runGit(t, fixture.repository, nil, "cat-file", "commit", "HEAD")
	header, _, found := bytes.Cut(raw, []byte("\n\n"))
	if !found {
		t.Fatal("fixture commit lacks its message separator")
	}
	malformed := append(slices.Clone(header), []byte("\n\nsubject\x00hidden completion\n")...)
	hash := strings.TrimSpace(string(runGit(
		t, fixture.repository, malformed,
		"hash-object", "-t", "commit", "-w", "--stdin", "--literally",
	)))

	_, err := gitCompletionMessages(t.Context(), fixture.repository, hash)
	if err == nil || !strings.Contains(err.Error(), "contains NUL") {
		t.Fatalf("raw commit NUL error = %v", err)
	}
}

func TestVerifyProspectiveCompletionTransition(t *testing.T) {
	parent := standardCompletionPlan()
	child, err := Advance(parent, "root", "do")
	if err != nil {
		t.Fatal(err)
	}
	manifest := testutil.ArtifactID(t, artifact.KindRecipe, "prospective-manifest")
	codeManifest := testutil.ArtifactID(t, artifact.KindProfile, "prospective-code-manifest")
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "prospective-preparation")
	preparationCommit := artifact.CommitID{1}
	message, err := CompletionCommitMessageWithMergeAuthority(
		[]byte("prospective completion"), parent, "root", "do",
		manifest, codeManifest, preparation, preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{parent}, nil, parent, child, string(message), MergeProjectionSemanticUnion,
	); err != nil {
		t.Fatalf("exact prospective transition refused: %v", err)
	}
	withAddedWork := parent
	withAddedWork.Items = slices.Clone(parent.Items)
	withAddedWork.Items = append(withAddedWork.Items, Item{
		ID: "new-work", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	})
	childWithAddedWork, err := Advance(withAddedWork, "root", "do")
	if err != nil {
		t.Fatal(err)
	}
	messageWithAddedWork, err := CompletionCommitMessageWithMergeAuthority(
		[]byte("prospective completion with added work"), withAddedWork, "root", "do",
		manifest, codeManifest, preparation, preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{parent}, nil, withAddedWork, childWithAddedWork, string(messageWithAddedWork), MergeProjectionSemanticUnion,
	); err != nil {
		t.Fatalf("prospective transition with added open work refused: %v", err)
	}
	baselineWithUnrelated := parent
	baselineWithUnrelated.Items = append(baselineWithUnrelated.Items, Item{ID: "empty", Status: StatusOpen})
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{baselineWithUnrelated}, nil, parent, child, string(message), MergeProjectionSemanticUnion,
	); err == nil || !strings.Contains(err.Error(), "deleted a baseline") {
		t.Fatalf("prospective unrelated deletion error = %v", err)
	}
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{parent, parent, parent}, nil, parent, child, string(message), MergeProjectionSemanticUnion,
	); err == nil || !strings.Contains(err.Error(), "one or two parents") {
		t.Fatalf("prospective octopus merge error = %v", err)
	}

	base := standardCompletionPlan()
	local := base
	local.Items = slices.Clone(base.Items)
	local.Items[0].Title = "local title"
	upstream := base
	upstream.Doctrine = "upstream doctrine"
	merged, err := MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	mergeChild, err := Advance(merged, "root", "do")
	if err != nil {
		t.Fatal(err)
	}
	mergeMessage, err := CompletionCommitMessageWithMergeAuthority(
		[]byte("prospective merge completion"), merged, "root", "do",
		manifest, codeManifest, preparation, preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{local, upstream}, &base, merged, mergeChild, string(mergeMessage), MergeProjectionSemanticUnion,
	); err != nil {
		t.Fatalf("exact prospective merge transition refused: %v", err)
	}

	completedMain, err := Advance(base, "root", "do")
	if err != nil {
		t.Fatal(err)
	}
	completedMainChild, err := Advance(completedMain, "dependent", "do")
	if err != nil {
		t.Fatal(err)
	}
	staleMergeMessage, err := CompletionCommitMessageWithMergeAuthority(
		[]byte("stale-source merge"), completedMain, "dependent", "do",
		manifest, codeManifest, preparation, preparationCommit, MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyProspectiveCompletionTransitionWithProjection(
		[]Plan{completedMain, base}, &base, completedMain, completedMainChild, string(staleMergeMessage), MergeProjectionSemanticUnion,
	); err == nil || !strings.Contains(err.Error(), "merge source must be rebased onto the current protected plan") {
		t.Fatalf("stale merge source policy error = %v", err)
	}
}

func TestProspectiveMergeAuthorityRequiresTargetCompletionEvidence(t *testing.T) {
	const repository = "test-repository"
	const localRevision = "1111111111111111111111111111111111111111"
	const incomingRevision = "2222222222222222222222222222222222222222"
	local := Plan{Items: []Item{{
		ID: "local", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	incoming := Plan{Items: []Item{{
		ID: "incoming", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	merged := Plan{Items: []Item{
		local.Items[0], incoming.Items[0], {
			ID: "waiting", Status: StatusOpen,
			Steps: []Step{{
				ID: "do", Status: StatusOpen, Verify: "go test ./...",
				DependsOn: []string{"retired/do"},
			}},
		},
	}}
	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, "common")
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "common")
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision,
		local, incoming, merged, localAuthority, incomingAuthority, MergeProjectionSemanticUnion,
	); err == nil || !strings.Contains(err.Error(), "pruned dependency retired/do") {
		t.Fatalf("missing target evidence error = %v", err)
	}
	incomingAuthority.completedReferences["retired/do"] = completionEvidence{commit: incomingRevision}
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision,
		local, incoming, merged, localAuthority, incomingAuthority, MergeProjectionSemanticUnion,
	); err != nil {
		t.Fatalf("target-authorized merge refused: %v", err)
	}
}

func TestProspectiveMergeAuthorityRejectsCrossParentIdentityReuse(t *testing.T) {
	const repository = "test-repository"
	const localRevision = "1111111111111111111111111111111111111111"
	const incomingRevision = "2222222222222222222222222222222222222222"
	local := Plan{Items: []Item{{
		ID: "reused", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	incoming := Plan{}
	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, "common")
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "common")
	incomingAuthority.completedReferences["reused/do"] = completionEvidence{commit: incomingRevision}
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision,
		local, incoming, local, localAuthority, incomingAuthority, MergeProjectionSemanticUnion,
	); err == nil || !strings.Contains(err.Error(), "reuses completed identity reused/do") {
		t.Fatalf("cross-parent identity reuse error = %v", err)
	}
	incomingAuthority.completedReferences = map[string]completionEvidence{}
	incomingAuthority.retiredItems["reused"] = completionEvidence{commit: incomingRevision, retiredItem: true}
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision,
		local, incoming, local, localAuthority, incomingAuthority, MergeProjectionSemanticUnion,
	); err == nil || !strings.Contains(err.Error(), "reuses retired item reused") {
		t.Fatalf("cross-parent retired item reuse error = %v", err)
	}
}

func TestProspectiveMergeAuthorityRejectsAmbiguousParentEvidence(t *testing.T) {
	const repository = "test-repository"
	const localRevision = "1111111111111111111111111111111111111111"
	const incomingRevision = "2222222222222222222222222222222222222222"
	local, incoming := Plan{}, Plan{}
	merged := Plan{Items: []Item{{
		ID: "waiting", Status: StatusOpen,
		Steps: []Step{{
			ID: "do", Status: StatusOpen, Verify: "go test ./...",
			DependsOn: []string{"retired/do"},
		}},
	}}}
	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, "common")
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "common")
	localAuthority.completedReferences["retired/do"] = completionEvidence{commit: localRevision}
	incomingAuthority.completedReferences["retired/do"] = completionEvidence{commit: incomingRevision}
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision,
		local, incoming, merged, localAuthority, incomingAuthority, MergeProjectionSemanticUnion,
	); err == nil || !strings.Contains(err.Error(), "ambiguous completion authority for retired/do") {
		t.Fatalf("ambiguous parent evidence error = %v", err)
	}
}

func TestProspectiveMergeAuthorityRequiresCommonProtectedEpoch(t *testing.T) {
	const repository = "test-repository"
	const localRevision = "1111111111111111111111111111111111111111"
	const incomingRevision = "2222222222222222222222222222222222222222"
	local, incoming, merged := Plan{}, Plan{}, Plan{}
	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, "local-seed")
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "incoming-seed")
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision,
		local, incoming, merged, localAuthority, incomingAuthority, MergeProjectionSemanticUnion,
	); err == nil || !strings.Contains(err.Error(), "do not share a protected completion epoch") {
		t.Fatalf("disjoint protected epoch error = %v", err)
	}
}

func TestProspectiveMergeAuthorityCanonicalizesRelativeRepository(t *testing.T) {
	repository, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	const localRevision = "1111111111111111111111111111111111111111"
	const incomingRevision = "2222222222222222222222222222222222222222"
	local, incoming, merged := Plan{}, Plan{}, Plan{}
	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, "common")
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "common")
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		".", localRevision, incomingRevision,
		local, incoming, merged, localAuthority, incomingAuthority, MergeProjectionSemanticUnion,
	); err != nil {
		t.Fatalf("relative repository refused: %v", err)
	}
}

func TestProspectiveMergeAuthorityBindsBothParents(t *testing.T) {
	const repository = "test-repository"
	const localRevision = "1111111111111111111111111111111111111111"
	const incomingRevision = "2222222222222222222222222222222222222222"
	local, incoming, merged := Plan{}, Plan{}, Plan{}
	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, "common")
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "common")
	other := Plan{Items: []Item{{
		ID: "other", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	for _, test := range []struct {
		name                        string
		requestedRepository         string
		requestedLocal, requestedIn string
		local, incoming             CompletionAuthority
	}{
		{name: "wrong repository", requestedRepository: repository + "-other", requestedLocal: localRevision, requestedIn: incomingRevision, local: localAuthority, incoming: incomingAuthority},
		{name: "stale local", requestedRepository: repository, requestedLocal: incomingRevision, requestedIn: incomingRevision, local: localAuthority, incoming: incomingAuthority},
		{name: "stale incoming", requestedRepository: repository, requestedLocal: localRevision, requestedIn: localRevision, local: localAuthority, incoming: incomingAuthority},
		{name: "cross-plan local", requestedRepository: repository, requestedLocal: localRevision, requestedIn: incomingRevision, local: prospectiveMergeAuthority(t, other, repository, localRevision, "common"), incoming: incomingAuthority},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := VerifyProspectiveMergeAuthorityWithProjection(
				test.requestedRepository, test.requestedLocal, test.requestedIn,
				local, incoming, merged, test.local, test.incoming, MergeProjectionSemanticUnion,
			); err == nil {
				t.Fatal("mismatched parent authority accepted")
			}
		})
	}
}

func prospectiveMergeAuthority(
	t *testing.T,
	document Plan,
	repository, revision, protectionSeed string,
) CompletionAuthority {
	t.Helper()
	authority := testCompletionAuthorityAt(t, document, repository, revision)
	authority.protectionSeeds = map[string]bool{protectionSeed: true}
	return authority
}

func TestProtectedCompletionCommitsIgnoreHistoryOrder(t *testing.T) {
	commits := []gitCompletionMessage{
		{hash: "merge", parents: []string{"left", "right"}},
		{hash: "unrelated-child", parents: []string{"unrelated"}},
		{hash: "seed", parents: []string{"old"}},
		{hash: "tip", parents: []string{"merge"}},
		{hash: "right", parents: []string{"seed"}},
		{hash: "old"},
		{hash: "left", parents: []string{"seed"}},
		{hash: "unrelated"},
	}
	protected, protectionSeeds := protectedCompletionCommits(commits, map[string]bool{"seed": true})
	got := make(map[string]bool, len(protected))
	for _, commit := range protected {
		got[commit.hash] = true
	}
	for _, want := range []string{"seed", "left", "right", "merge", "tip"} {
		if !got[want] {
			t.Errorf("skewed merge descendant %s escaped protected history", want)
		}
	}
	for _, unwanted := range []string{"old", "unrelated", "unrelated-child"} {
		if got[unwanted] {
			t.Errorf("pre-activation or unrelated commit %s became protected", unwanted)
		}
	}
	if !protectionSeeds["merge"]["seed"] || !protectionSeeds["tip"]["seed"] {
		t.Fatalf("prepared seed provenance did not cross the skewed merge DAG: %v", protectionSeeds)
	}
}

func TestProtectedMergeRequiresRebasedSourceHistory(t *testing.T) {
	fixture := newCompletionFixture(t, standardCompletionPlan(), "activation", "do")
	fixture.preAdvance.Items = append(slices.Clone(fixture.parent.Items), Item{
		ID: "activation", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: fixture.verify}},
	})
	root := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
	rootTree := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD^{tree}")))
	transient := Item{
		ID: "transient-x", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}
	withTransient := fixture.parent
	withTransient.Items = append(slices.Clone(withTransient.Items), transient)
	if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), withTransient); err != nil {
		t.Fatal(err)
	}
	runGit(t, fixture.repository, nil, "add", "--", Path)
	transientTree := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "write-tree")))
	sideAdd := commitFixtureTree(t, fixture.repository, []byte("add transient row\n"), transientTree, root)
	sideDelete := commitFixtureTree(t, fixture.repository, []byte("raw-delete transient row\n"), rootTree, sideAdd)

	fixture.commit(fixture.canonicalMessage(), true)
	merge := commitFixtureTree(
		t, fixture.repository, []byte("merge unreconciled side history\n"), rootTree,
		fixture.completionHash, sideDelete,
	)
	runGit(t, fixture.repository, nil, "update-ref", "HEAD", merge)
	reused := fixture.child
	reused.Items = append(slices.Clone(reused.Items), transient)
	if _, err := resolveFixture(fixture, reused, "HEAD"); err == nil ||
		!strings.Contains(err.Error(), "rebase the merge source onto the protected plan") {
		t.Fatalf("unprotected transient side history error = %v", err)
	}
}

func TestProtectedMergeRequiresCommonPreparedAncestor(t *testing.T) {
	fixture := newCompletionFixture(t, standardCompletionPlan(), "main-activation", "do")
	fixture.preAdvance.Items = append(slices.Clone(fixture.parent.Items), Item{
		ID: "main-activation", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: fixture.verify}},
	})
	root := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
	rootTree := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD^{tree}")))
	transient := Item{
		ID: "transient-x", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}
	withTransient := fixture.parent
	withTransient.Items = append(slices.Clone(withTransient.Items), transient)
	if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), withTransient); err != nil {
		t.Fatal(err)
	}
	runGit(t, fixture.repository, nil, "add", "--", Path)
	transientTree := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "write-tree")))
	sideAdd := commitFixtureTree(t, fixture.repository, []byte("side transient add\n"), transientTree, root)
	sideDelete := commitFixtureTree(t, fixture.repository, []byte("side transient delete\n"), rootTree, sideAdd)

	fixture.commit(fixture.canonicalMessage(), true)
	mainSeed := fixture.completionHash
	fixture.item, fixture.step = "side-activation", "do"
	fixture.preAdvance = fixture.parent
	fixture.preAdvance.Items = append(slices.Clone(fixture.parent.Items), Item{
		ID: "side-activation", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: fixture.verify}},
	})
	fixture.child = fixture.parent
	fixture.rotatePreparation(strings.Repeat("f", 64), time.Unix(5, 0))
	fixture.manifest = testutil.ArtifactID(t, artifact.KindRecipe, "side-seed-manifest")
	fixture.codeManifest = testutil.ArtifactID(t, artifact.KindProfile, "side-seed-code-manifest")
	sideSeed := commitFixtureTree(
		t, fixture.repository, fixture.canonicalMessage(), rootTree, sideDelete,
	)
	publishCompletionAttempt(
		t, fixture.store, sideSeed, fixture.item, fixture.step, fixture.verify, fixture.acceptance,
		fixture.manifest, fixture.codeManifest, fixture.preparation,
	)
	merge := commitFixtureTree(
		t, fixture.repository, []byte("merge independently activated history\n"), rootTree,
		mainSeed, sideSeed,
	)
	runGit(t, fixture.repository, nil, "update-ref", "HEAD", merge)
	reused := fixture.parent
	reused.Items = append(slices.Clone(reused.Items), transient)
	if _, err := resolveFixture(fixture, reused, "HEAD"); err == nil ||
		!strings.Contains(err.Error(), "has no common prepared ancestor") {
		t.Fatalf("independent prepared seeds were merged: %v", err)
	}
}

func TestCompletedPlanIdentityCannotBeReused(t *testing.T) {
	t.Run("retired step and item", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		for name, row := range map[string]Item{
			"same step": {ID: "root", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
			"new step":  {ID: "root", Status: StatusOpen, Steps: []Step{{ID: "again", Status: StatusOpen, Verify: "go test ./..."}}},
		} {
			t.Run(name, func(t *testing.T) {
				reused := fixture.child
				reused.Items = append(reused.Items, row)
				if _, err := resolveFixture(fixture, reused, "HEAD"); err == nil || !strings.Contains(err.Error(), "cannot be reused") {
					t.Fatalf("reuse error = %v", err)
				}
			})
		}
		foreign := fixture.child
		foreign.Items = append(foreign.Items, Item{
			ID: "other", Status: StatusOpen,
			Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
		})
		if _, err := resolveFixture(fixture, foreign, "HEAD"); err != nil {
			t.Fatalf("same step under a new item refused: %v", err)
		}
	})
	t.Run("multi step item remains live", func(t *testing.T) {
		parent := standardCompletionPlan()
		parent.Items[0].Steps = append(parent.Items[0].Steps, Step{
			ID: "later", Status: StatusOpen, Verify: "go test ./...",
		})
		fixture := newCompletionFixture(t, parent, "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err != nil {
			t.Fatalf("remaining item step refused: %v", err)
		}
	})
	t.Run("second completion is ambiguous", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		fixture.manifest = testutil.ArtifactID(t, artifact.KindRecipe, "second-manifest")
		fixture.codeManifest = testutil.ArtifactID(t, artifact.KindProfile, "second-code-manifest")
		message := fixture.canonicalMessage()
		runGit(t, fixture.repository, message, "commit", "-q", "--allow-empty", "-F", "-")
		second := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
		publishCompletionAttempt(
			t, fixture.store, second, fixture.item, fixture.step, fixture.verify, fixture.acceptance,
			fixture.manifest, fixture.codeManifest, fixture.preparation,
		)
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil {
			t.Fatalf("repeated no-op completion error = %v", err)
		}
	})
	t.Run("prepared identity is global without a surviving dependency", func(t *testing.T) {
		parent := Plan{Campaign: "completion", Doctrine: "fixture", Items: []Item{{
			ID: "root", Status: StatusOpen,
			Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
		}}}
		fixture := newCompletionFixture(t, parent, "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)

		if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), parent); err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, nil, "add", "--", Path)
		runGit(t, fixture.repository, nil, "commit", "-q", "-m", "re-add completed identity")

		fixture.rotatePreparation(strings.Repeat("b", 64), time.Unix(2, 0))
		fixture.preAdvance = parent
		fixture.manifest = testutil.ArtifactID(t, artifact.KindRecipe, "second-global-manifest")
		fixture.codeManifest = testutil.ArtifactID(t, artifact.KindProfile, "second-global-code-manifest")
		fixture.commit(fixture.canonicalMessage(), true)

		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "is ambiguous") {
			t.Fatalf("globally repeated prepared identity error = %v", err)
		}
	})
}

func TestCompletionAuthorityDerivesMergeTransition(t *testing.T) {
	fixture := newCompletionFixture(t, standardCompletionPlan(), "activation", "do")
	fixture.preAdvance.Items = append(slices.Clone(fixture.parent.Items), Item{
		ID: "activation", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: fixture.verify}},
	})
	fixture.commit(fixture.canonicalMessage(), true)
	primary := strings.TrimSpace(string(runGit(
		t, fixture.repository, nil, "rev-parse", "--abbrev-ref", "HEAD",
	)))
	runGit(t, fixture.repository, nil, "checkout", "-q", "-b", "completion-side")
	runGit(t, fixture.repository, nil, "commit", "-q", "--allow-empty", "-m", "side parent")
	runGit(t, fixture.repository, nil, "checkout", "-q", primary)
	runGit(t, fixture.repository, nil, "commit", "-q", "--allow-empty", "-m", "first parent")
	runGit(t, fixture.repository, nil, "merge", "--no-ff", "--no-commit", "completion-side")
	fixture.item, fixture.step = "root", "do"
	fixture.verify, fixture.acceptance = "go test ./...", "go test ./..."
	fixture.preAdvance = fixture.parent
	var err error
	fixture.child, err = Advance(fixture.parent, fixture.item, fixture.step)
	if err != nil {
		t.Fatal(err)
	}
	fixture.rotatePreparation(strings.Repeat("e", 64), time.Unix(4, 0))
	fixture.manifest = testutil.ArtifactID(t, artifact.KindRecipe, "merge-manifest")
	fixture.codeManifest = testutil.ArtifactID(t, artifact.KindProfile, "merge-code-manifest")
	if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), fixture.child); err != nil {
		t.Fatal(err)
	}
	runGit(t, fixture.repository, nil, "add", "--", Path)
	runGit(t, fixture.repository, fixture.canonicalMessage(), "commit", "-q", "-F", "-")
	fixture.completionHash = strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
	parents := strings.Fields(string(runGit(
		t, fixture.repository, nil, "rev-list", "--parents", "-n", "1", fixture.completionHash,
	)))
	if len(parents) != 3 {
		t.Fatalf("completion fixture is not a two-parent merge: %q", parents)
	}
	publishCompletionAttempt(
		t, fixture.store, fixture.completionHash, fixture.item, fixture.step,
		fixture.verify, fixture.acceptance, fixture.manifest, fixture.codeManifest, fixture.preparation,
	)
	if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err != nil {
		t.Fatal(err)
	}
}

func TestCompletionAuthoritySelectsTargetMergeBase(t *testing.T) {
	fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
	root := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
	rootTree := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD^{tree}")))
	left := commitFixtureTree(t, fixture.repository, []byte("left\n"), rootTree, root)
	right := commitFixtureTree(t, fixture.repository, []byte("right\n"), rootTree, root)
	leftMerge := commitFixtureTree(t, fixture.repository, []byte("left merge\n"), rootTree, left, right)
	rightMerge := commitFixtureTree(t, fixture.repository, []byte("right merge\n"), rootTree, right, left)
	bases := strings.Fields(string(runGit(
		t, fixture.repository, nil, "merge-base", "--all", leftMerge, rightMerge,
	)))
	if len(bases) != 2 {
		t.Fatalf("criss-cross fixture merge bases = %v, want two", bases)
	}

	if err := Save(filepath.Join(fixture.repository, filepath.FromSlash(Path)), fixture.child); err != nil {
		t.Fatal(err)
	}
	runGit(t, fixture.repository, nil, "add", "--", Path)
	childTree := strings.TrimSpace(string(runGit(t, fixture.repository, nil, "write-tree")))
	fixture.completionHash = commitFixtureTree(
		t, fixture.repository, fixture.canonicalMessage(), childTree, leftMerge, rightMerge,
	)
	transitions, err := completionTransitionPlans(t.Context(), fixture.repository, []gitCompletionMessage{{
		hash: fixture.completionHash, parents: []string{leftMerge, rightMerge},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got := transitions[fixture.completionHash].mergeBaseRevision; got != left {
		t.Fatalf("replayed merge base = %q, want target base %q", got, left)
	}
	if got, err := CompletionMergeBase(t.Context(), fixture.repository, rightMerge, leftMerge); err != nil || got != right {
		t.Fatalf("reversed target merge base = %q, %v; want %q", got, err, right)
	}
	fork := commitFixtureTree(t, fixture.repository, []byte("fork\n"), rootTree, root)
	forkMerge := commitFixtureTree(t, fixture.repository, []byte("fork merge\n"), rootTree, fork, leftMerge)
	_, err = CompletionMergeBase(t.Context(), fixture.repository, forkMerge, rightMerge)
	if err == nil || !strings.Contains(err.Error(), "first-parent chain, found 0 of 2") {
		t.Fatalf("ambiguous completion merge-base error = %v", err)
	}
}

func commitFixtureTree(t *testing.T, repository string, message []byte, tree string, parents ...string) string {
	t.Helper()
	arguments := []string{"commit-tree", tree}
	for _, parent := range parents {
		arguments = append(arguments, "-p", parent)
	}
	return strings.TrimSpace(string(runGit(t, repository, message, arguments...)))
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
