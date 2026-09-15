package gate

import (
	"errors"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// batchGateInvocations binds the fixture's batch with the cumulative checks
// a gate carries after acceptance, the dependent suite and the commit, whose
// bodies are the caller's: a refusal proves a focused publication deferred
// them, a pass models the cumulative gate.
func batchGateInvocations(t *testing.T, g *gateContext, batch *plan.VerificationBatch, tree string, cumulative func() (bool, error)) ([]automationcheck.Check, []automationcheck.Invocation) {
	t.Helper()
	rest := gateCheck(testRestCheckName, runrecord.PhaseTest, cumulative)
	rest.Descriptor.Dependencies = []string{"acceptance"}
	commit := gateCheck("commit", runrecord.PhasePackage, cumulative)
	commit.Descriptor.Dependencies = []string{testRestCheckName}
	checks, err := g.batchAcceptanceChecks([]automationcheck.Check{
		gateCheck("acceptance", runrecord.PhaseTest, g.stepAcceptance), rest, commit,
	}, batch)
	if err != nil {
		t.Fatal(err)
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := automationcheck.BindManifestPlan(
		testutil.ArtifactID(t, artifact.KindProfile, "repair base"),
		testutil.ArtifactID(t, artifact.KindProfile, "repair candidate"),
		strings.Repeat("a", 64), candidateTreeKey(tree),
		automationcheck.Surface{Identity: "repair batch fixture"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	g.manifestPlan = &manifest
	for index, invocation := range invocations {
		invocations[index], err = g.bindCheckExecution(manifest, invocation, manifest.CandidateManifest)
		if err != nil {
			t.Fatal(err)
		}
	}
	return checks, invocations
}

func refuseCumulative() (bool, error) {
	return false, errors.New("a cumulative check ran during a focused checkpoint publication")
}

func passCumulative() (bool, error) { return false, nil }

// TestGateRepairBatchAcceptance holds focused checkpoint publication on the
// batch fixture: a publication runs its checkpoint and the declared
// predecessors alone and defers the step acceptance, the dependent suite,
// the commit and every other checkpoint without satisfying them; an
// unrelated sibling failure never runs; a relevant failure is refused while
// the sibling's receipt stays retained; changed inputs execute again; the
// cumulative gate reuses retained receipts; the ledger survives a store
// reopen; and the completion lands through the commit path. Restart and
// cancellation of the same ledger are held by TestCheckpointPersistenceSurvivesKill
// and TestTerminalEvidenceRestart.
func TestGateRepairBatchAcceptance(t *testing.T) {
	t.Parallel()
	g, batch, tree := verificationBatchFixture(t, "failure")
	g.checkpoint = "producer"
	checks, invocations := batchGateInvocations(t, g, batch, tree, refuseCumulative)
	deferred, err := g.checkpointDeferrals(checks)
	if err != nil {
		t.Fatal(err)
	}
	if want := map[string]bool{"acceptance": true, testRestCheckName: true, "commit": true, "acceptance-consumer": true}; !maps.Equal(deferred, want) {
		t.Fatalf("focused producer deferred %v, want %v", deferred, want)
	}
	cache := g.loadRetryCache()
	results, err := g.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil, deferred)
	if err != nil || len(results) != 1 || results[0].Invocation.Check.Name != "acceptance-producer" || results[0].Err != nil || !results[0].Evidence.ID.Valid() {
		t.Fatalf("focused producer publication: %+v %v", results, err)
	}
	ledger, err := g.openBatchEvidence(invocations, nil, &cache)
	if err != nil {
		t.Fatal(err)
	}
	if producer, consumer := ledger.state.Checks["acceptance-producer"], ledger.state.Checks["acceptance-consumer"]; !producer.Result.Evidence.Valid() ||
		consumer.Result.Evidence.Valid() || consumer.Obligation.Kind() != artifact.KindEvidence {
		t.Fatalf("ledger after the focused publication: producer=%+v consumer=%+v", producer, consumer)
	}
	if err := g.closeStore(); err != nil {
		t.Fatal(err)
	}
	if _, err := focusedCheckpoints(batch, "absent"); err == nil {
		t.Fatal("an undeclared checkpoint was admitted")
	}

	// A focused consumer: its relevant failure is refused; the producer it
	// declares as predecessor is reused from the retained receipt.
	second, batch, sameTree := verificationBatchContext(t, g.repo)
	if sameTree != tree {
		t.Fatal("unchanged fixture changed its planned tree")
	}
	second.checkpoint = "consumer"
	checks, invocations = batchGateInvocations(t, second, batch, tree, refuseCumulative)
	deferred, err = second.checkpointDeferrals(checks)
	if err != nil {
		t.Fatal(err)
	}
	if want := map[string]bool{"acceptance": true, testRestCheckName: true, "commit": true}; !maps.Equal(deferred, want) {
		t.Fatalf("focused consumer deferred %v, want %v", deferred, want)
	}
	cache = second.loadRetryCache()
	results, err = second.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil, deferred)
	if err != nil || len(results) != 2 {
		t.Fatalf("focused consumer publication: %+v %v", results, err)
	}
	byName := map[string]automationcheck.DAGResult{}
	for _, result := range results {
		byName[result.Invocation.Check.Name] = result
	}
	if !byName["acceptance-producer"].Evidence.Reused || byName["acceptance-producer"].Err != nil {
		t.Fatalf("producer receipt was not reused: %+v", byName["acceptance-producer"])
	}
	if byName["acceptance-consumer"].Err == nil {
		t.Fatal("relevant consumer failure was admitted")
	}
	ledger, err = second.openBatchEvidence(invocations, nil, &cache)
	if err != nil {
		t.Fatal(err)
	}
	if !ledger.state.Checks["acceptance-producer"].Result.Evidence.Valid() || ledger.state.Checks["acceptance-consumer"].Result.Evidence.Valid() {
		t.Fatal("relevant failure changed the retained sibling receipt or gained credit")
	}
	if err := second.closeStore(); err != nil {
		t.Fatal(err)
	}

	// The repair changes an input of both checkpoints: the cumulative gate
	// executes the producer again instead of reusing the retained receipt.
	if err := os.WriteFile(filepath.Join(g.repo, "mode"), []byte("pass"), 0o644); err != nil {
		t.Fatal(err)
	}
	third, batch, repairedTree := verificationBatchContext(t, g.repo)
	if repairedTree == tree {
		t.Fatal("repair left the planned tree unchanged")
	}
	checks, invocations = batchGateInvocations(t, third, batch, repairedTree, passCumulative)
	if deferred, err := third.checkpointDeferrals(checks); err != nil || len(deferred) != 0 {
		t.Fatalf("cumulative gate deferred %v: %v", deferred, err)
	}
	cache = third.loadRetryCache()
	results, err = third.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil, nil)
	if err != nil || len(results) != len(invocations) {
		t.Fatalf("cumulative gate: %+v %v", results, err)
	}
	var records []runrecord.GateStep
	for _, result := range results {
		if result.Err != nil || !result.Evidence.ID.Valid() {
			t.Fatalf("cumulative gate check %s: %+v", result.Invocation.Check.Name, result)
		}
		if result.Invocation.Check.Name == "acceptance-producer" && result.Evidence.Reused {
			t.Fatal("changed inputs reused the retained producer receipt")
		}
		records = append(records, gateEvidenceRecord(result.Invocation.Check.Name, result.Invocation.Check.Phase, result.Evidence, result.Err, third.stepEvidence[result.Invocation.Check.Name]))
	}
	if err := third.closeStore(); err != nil {
		t.Fatal(err)
	}

	// A further gate over the same candidate reuses both retained receipts.
	fourth, batch, _ := verificationBatchContext(t, g.repo)
	_, invocations = batchGateInvocations(t, fourth, batch, repairedTree, passCumulative)
	cache = fourth.loadRetryCache()
	results, err = fourth.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range results {
		if name := result.Invocation.Check.Name; strings.HasPrefix(name, "acceptance-") && (!result.Evidence.Reused || result.Err != nil) {
			t.Fatalf("sibling %s was not reused at promotion: %+v", name, result)
		}
	}
	if err := fourth.closeStore(); err != nil {
		t.Fatal(err)
	}

	// The ledger is durable: a read-only reopen of the store resolves both
	// terminal results under the batch alias.
	task, err := artifact.JSONID(artifact.KindRecipe, struct{ Policy, Reference string }{"gate-batch/v1", "audio/dataset"})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(g.repo, g.storePath))
	if err != nil {
		t.Fatal(err)
	}
	retained, found, err := batchEvidenceCodec.Resolve(t.Context(), store, "gate/batch/"+task.String())
	closeErr := store.Close()
	if err != nil || !found || closeErr != nil {
		t.Fatalf("reopen batch ledger: found=%v %v %v", found, err, closeErr)
	}
	for _, name := range []string{"acceptance-producer", "acceptance-consumer"} {
		if !retained.Checks[name].Result.Evidence.Valid() {
			t.Fatalf("reopened ledger lost %s", name)
		}
	}

	// The completion lands through the commit path over the reused members
	// and enters protected history; the step is no longer dispatchable.
	document, err := plan.Load(filepath.Join(g.repo, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	advanced, err := plan.Advance(document, "audio", "dataset")
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(candidateTreeKey(repairedTree), third.environment.ID, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join("tmp", "completion-store")
	preparationCommit := publishInterruptedPreparation(t, g.repo, storePath, third.environment, preparation)
	message, err := plan.CompletionCommitMessageWithMergeAuthority(
		[]byte("Complete focused checkpoint fixture"), document, "audio", "dataset",
		third.manifestPlan.ID, third.manifestPlan.CandidateManifest, preparation.ID, preparationCommit,
		plan.MergeProjectionSemanticUnion, artifact.ID{},
	)
	if err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(t.TempDir(), "message.txt")
	if err := os.WriteFile(messagePath, message, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(filepath.Join(g.repo, plan.Path), advanced); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, g.repo, "add", "--", ".")
	runGitFixture(t, g.repo, "commit", "-q", "-F", messagePath)
	fixture := interruptedCommitFixture{
		repo: g.repo, storePath: storePath, commit: recoveryGit(t, g.repo, "rev-parse", "HEAD"),
		recipe: third.manifestPlan.ID, candidate: third.manifestPlan.CandidateManifest,
		environment: third.environment, preparation: preparation, preparationCommit: preparationCommit,
		manifest: *third.manifestPlan,
	}
	members := slices.DeleteFunc(slices.Clone(records), func(step runrecord.GateStep) bool {
		return step.Name == "acceptance" || step.Name == "commit"
	})
	_, publication := successfulAttemptBatchForReference(t, fixture, "audio", "dataset",
		document.Items[0].Steps[0].Verify, true, "focused-checkpoint-completion", members...)
	completion := openRecoveryStore(t, fixture)
	_, publishErr := completion.Commit(t.Context(), publication)
	if publishErr != nil {
		completion.Close()
		t.Fatal(publishErr)
	}
	authority, err := plan.ResolveCompletionAuthority(t.Context(), g.repo, "HEAD", advanced, completion)
	closeErr = completion.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("resolve completion after the commit path: %v %v", err, closeErr)
	}
	if !authority.ProtectsRevision() {
		t.Fatal("completed fixture did not enter protected history")
	}
	if _, _, open := plan.Current(advanced, "", authority); open {
		t.Fatal("completed fixture task remained dispatchable")
	}
}
