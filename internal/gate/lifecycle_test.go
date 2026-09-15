package gate

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

func requireNoPendingGateState(repo, storePath string) error {
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		return fmt.Errorf("gate: inspect lifecycle authority: %w", err)
	}
	defer store.Close()
	return requireNoPendingGateStateWithStore(repo, store)
}

func TestGateDebtReconciliation(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "cgo=0", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := runrecord.NewGatePreparation(
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", environment.ID, time.Unix(100, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparationContent, _ := prepared.Content()
	environmentContent, _ := environment.Content()
	prepareBatch, err := artifact.NewDocumentBatch(
		"test/prepared", []artifact.Content{environmentContent, preparationContent}, prepared.Lineage(),
		[]artifact.AliasBinding{{Name: runrecord.GateLifecycleCurrentAlias, Target: prepared.ID}},
	)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), prepareBatch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	batch, _ := lifecycleDebtBatch(t, prepared, environment, "reconcile")
	g := gateContext{repo: repo, preparation: prepared}
	if err := g.oweRecord(batch, errors.New("injected OvergoDB outage")); err == nil {
		t.Fatal("oweRecord hid the triggering failure")
	}
	if got, err := reconcileGateDebt(repo, storePath); err != nil || got != prepared.ID {
		t.Fatalf("reconcile = (%s, %v)", got, err)
	}
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("debt payload remains after reconciliation: %v", err)
	}
	// A crash after the exact store append but before removing the debt file is
	// an idempotent replay, not a conflicting second finalization.
	if err := g.oweRecord(batch, errors.New("injected post-append crash")); err == nil {
		t.Fatal("oweRecord hid the replay trigger")
	}
	if got, err := reconcileGateDebt(repo, storePath); err != nil || got != prepared.ID {
		t.Fatalf("idempotent reconcile = (%s, %v)", got, err)
	}
}

func TestGateDebtReconciliationRefusesLaterUnaliasedFinalization(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	environment := lifecycleTestEnvironment(t)
	preparation, err := runrecord.NewGatePreparation(
		"abababababababababababababababababababababababababababababababab",
		environment.ID,
		time.Unix(101, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = publishInterruptedPreparation(t, repo, storePath, environment, preparation)
	debtBatch, debtFinalization := lifecycleDebtBatch(t, preparation, environment, "stale-reconcile")
	g := gateContext{repo: repo, preparation: preparation}
	if err := g.oweRecord(debtBatch, errors.New("injected OvergoDB outage")); err == nil {
		t.Fatal("oweRecord hid the triggering failure")
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	existing := publishUnaliasedGateFinalization(t, store, preparation, environment, "stale-reconcile-conflict")
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := reconcileGateDebt(repo, storePath); err == nil ||
		!strings.Contains(err.Error(), "another finalization") {
		t.Fatalf("stale debt reconciliation = %v", err)
	}
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); err != nil {
		t.Fatalf("conflicting debt was not preserved: %v", err)
	}
	store, err = overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	finalization, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, preparation.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || finalization.ID != existing.ID || finalization.ID == debtFinalization.ID {
		t.Fatalf("stale debt changed finalization authority: (%+v, %t)", finalization, found)
	}
}

func TestPreparedLifecycleLocatorBlocksAndRecovers(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, storePath := newLifecycleRepo(t), "store"
	t.Setenv("OVERGO_STRATEGY_ID", "")
	if err := os.WriteFile(filepath.Join(repo, "candidate.go"), []byte("package candidate\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "lifecycle@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Lifecycle Test")
	runGitFixture(t, repo, "add", "--", "candidate.go")
	runGitFixture(t, repo, "commit", "-q", "-m", "baseline")
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	g := gateContext{
		repo: repo, storePath: storePath, start: time.Unix(200, 0), environment: environment,
	}
	if err := g.prepare(); err != nil {
		t.Fatal(err)
	}
	if !g.preparationCommit.Valid() {
		t.Fatal("preparation has no durable commit receipt")
	}
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); err != nil {
		t.Fatal(err)
	}
	if heartbeat.State != runrecord.HeartbeatRunning || heartbeat.Preparation != g.preparation.ID {
		t.Fatalf("preparation locator = %+v", heartbeat)
	}
	if err := os.Remove(filepath.Join(repo, filepath.FromSlash(gateHeartbeatFile))); err != nil {
		t.Fatal(err)
	}
	if err := requireNoPendingGateState(repo, storePath); err == nil {
		t.Fatal("new gate admitted over an unresolved prepared lifecycle")
	}
	if got, err := recordSelectedUnbatchableFailure(repo, storePath, ""); err != nil || got != g.preparation.ID {
		t.Fatalf("recover prepared lifecycle = (%s, %v)", got, err)
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(t.Context(), store, g.preparation.ID)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if !found || finalization.Outcome != runrecord.OutcomeCancelled || finalization.Result == nil {
		store.Close()
		t.Fatalf("prepared lifecycle finalization = (%+v, %t)", finalization, found)
	}
	gate, err := runrecord.RequireGateResult(t.Context(), store, *finalization.Result)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if len(gate.Steps) != 1 || gate.Steps[0].DurationNS <= 1 {
		store.Close()
		t.Fatalf("record recovery retained a synthetic duration: %+v", gate.Steps)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	// A crash after the final append but before the locator update is
	// idempotently repaired instead of appending a second finalization.
	heartbeat.State = runrecord.HeartbeatRunning
	if err := writeJSON(repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := recordSelectedUnbatchableFailure(repo, storePath, ""); err != nil {
		t.Fatal(err)
	}
	if err := requireNoPendingGateState(repo, storePath); err != nil {
		t.Fatalf("finalized lifecycle still blocks the gate: %v", err)
	}
}

func TestUncommittedLifecycleLocatorIsRemoved(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(
		"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		environment.ID,
		time.Unix(300, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: repo, environment: environment, preparation: preparation}
	if err := g.writeHeartbeat(runrecord.HeartbeatRunning); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(repo, storePath, ""); err != nil || got != preparation.ID {
		t.Fatalf("recover uncommitted locator = (%s, %v)", got, err)
	}
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateHeartbeatFile))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("uncommitted locator remains: %v", err)
	}
}

func TestUncommittedLifecycleLocatorIsRemovedAfterPriorFinalization(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	ctx := t.Context()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	previous, err := runrecord.NewGatePreparation(
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		environment.ID,
		time.Unix(300, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	previousContent, err := previous.Content()
	if err != nil {
		t.Fatal(err)
	}
	resultID, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("prior finalized result"))
	if err != nil {
		t.Fatal(err)
	}
	finalized, err := runrecord.NewGateFinalization(
		previous, "0123456789abcdef0123456789abcdef01234567", resultID, runrecord.OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizedContent, err := finalized.Content()
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"test/prior-finalized-lifecycle",
		[]artifact.Content{environmentContent, previousContent, finalizedContent},
		append(previous.Lineage(), finalized.Lineage()...),
		[]artifact.AliasBinding{{Name: runrecord.GateLifecycleCurrentAlias, Target: finalized.ID}},
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: resultID})
	if _, err := store.Commit(ctx, batch); err != nil {
		store.Close()
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	uncommitted, err := runrecord.NewGatePreparation(
		"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		environment.ID,
		time.Unix(301, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: repo, environment: environment, preparation: uncommitted}
	if err := g.writeHeartbeat(runrecord.HeartbeatRunning); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(repo, storePath, ""); err != nil || got != uncommitted.ID {
		t.Fatalf("recover uncommitted locator after prior finalization = (%s, %v)", got, err)
	}
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateHeartbeatFile))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("uncommitted locator remains: %v", err)
	}
	store, err = overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != finalized.ID {
		t.Fatalf("prior finalized authority changed: found=%t current=%s want=%s", found, current, finalized.ID)
	}
}

func TestFinalizedLifecycleAliasDoesNotHideUnaliasedPreparation(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, environment := newCompleteLifecycleAliasFixture(t)
	later, err := runrecord.NewGatePreparation(
		"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		environment.ID,
		time.Unix(401, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	laterContent, err := later.Content()
	if err != nil {
		t.Fatal(err)
	}
	laterBatch, err := artifact.NewDocumentBatch(
		"test/finalized-alias/unaliased-preparation",
		[]artifact.Content{laterContent},
		later.Lineage(),
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, laterBatch); err != nil {
		t.Fatal(err)
	}
	if err := requireNoOutstandingGateLifecycle(ctx, store); err == nil || !strings.Contains(err.Error(), "1 unresolved") {
		t.Fatalf("final alias with unaliased preparation = %v", err)
	}
	if _, err := gatePreparationAlias(ctx, store, later.ID); err == nil || !strings.Contains(err.Error(), "1 unresolved") {
		t.Fatalf("preparation alias admitted over unaliased debt = %v", err)
	}
}

func TestRecordFailureClosesUniqueStalePreparationBehindFinalizedAuthority(t *testing.T) {
	t.Parallel()
	fixture := newStaleLifecycleRecoveryFixture(t, 1, true)
	heartbeatBefore, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(gateHeartbeatFile)))
	if err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err != nil || got != fixture.stale[0].ID {
		t.Fatalf("recover stale preparation = (%s, %v)", got, err)
	}
	heartbeatAfter, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(gateHeartbeatFile)))
	if err != nil {
		t.Fatal(err)
	}
	if string(heartbeatAfter) != string(heartbeatBefore) {
		t.Fatal("stale-debt recovery replaced the newer finalized heartbeat")
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.currentFinalization.ID {
		t.Fatalf("current lifecycle authority = (%s, %t), want preserved %s", current, found, fixture.currentFinalization.ID)
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(ctx, store, fixture.stale[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	if !found || finalization.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("stale preparation finalization = (%+v, %t), want cancellation", finalization, found)
	}
	if err := validateCompleteGateFinalization(ctx, store, fixture.stale[0], finalization); err != nil {
		t.Fatalf("stale cancellation is incomplete: %v", err)
	}
	if err := requireNoOutstandingGateLifecycle(ctx, store); err != nil {
		t.Fatalf("stale cancellation left lifecycle debt: %v", err)
	}
}

func TestRecordFailureWithFinalizedAliasRefusesPostCensusUnaliasedPreparation(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	fixture := newStaleLifecycleRecoveryFixture(t, 1, true)
	previousHook := gateRecordFailureBeforeStoreCommitHook
	t.Cleanup(func() { gateRecordFailureBeforeStoreCommitHook = previousHook })
	var racingPreparation runrecord.GateLifecycle
	gateRecordFailureBeforeStoreCommitHook = func(store *overgodb.Store) {
		racingPreparation = publishLegacyGatePreparation(
			t, store, lifecycleTestEnvironment(t), strings.Repeat("9", 64), time.Unix(629, 0),
			"finalized-alias-head-race",
		)
	}
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!errors.Is(err, artifact.ErrCommitPrecondition) {
		t.Fatalf("finalized-alias lifecycle head race = %v", err)
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.currentFinalization.ID {
		t.Fatalf("head race moved finalized alias to %s", current)
	}
	for _, preparation := range []runrecord.GateLifecycle{fixture.stale[0], racingPreparation} {
		if _, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID); err != nil {
			t.Fatal(err)
		} else if found {
			t.Fatalf("head race published finalization for %s", preparation.ID)
		}
	}
}

func TestRecordFailureWithPreparedAliasAndNoHeartbeatRefusesPostCensusPreparation(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo, storePath := newLifecycleRepo(t), "store"
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "prepared-alias-race@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Prepared Alias Race Test")
	runGitFixture(t, repo, "commit", "--allow-empty", "-q", "-m", "baseline")
	environment := lifecycleTestEnvironment(t)
	preparation, err := runrecord.NewGatePreparation(
		strings.Repeat("c", 64), environment.ID, time.Unix(631, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	publishInterruptedPreparation(t, repo, storePath, environment, preparation)
	previousHook := gateRecordFailureBeforeStoreCommitHook
	t.Cleanup(func() { gateRecordFailureBeforeStoreCommitHook = previousHook })
	var racingPreparation runrecord.GateLifecycle
	gateRecordFailureBeforeStoreCommitHook = func(store *overgodb.Store) {
		racingPreparation = publishLegacyGatePreparation(
			t, store, environment, strings.Repeat("d", 64), time.Unix(632, 0),
			"prepared-alias-head-race",
		)
	}
	if _, err := recordSelectedUnbatchableFailure(repo, storePath, ""); err == nil ||
		!errors.Is(err, artifact.ErrCommitPrecondition) {
		t.Fatalf("prepared-alias lifecycle head race = %v", err)
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != preparation.ID {
		t.Fatalf("prepared-alias head race moved authority to %s", current)
	}
	for _, candidate := range []runrecord.GateLifecycle{preparation, racingPreparation} {
		if _, found, err := runrecord.GateFinalizationForPreparation(ctx, store, candidate.ID); err != nil {
			t.Fatal(err)
		} else if found {
			t.Fatalf("prepared-alias head race finalized %s", candidate.ID)
		}
	}
}

func TestRecordFailureRefusesAmbiguousStalePreparationsBehindFinalizedAuthority(t *testing.T) {
	t.Parallel()
	fixture := newStaleLifecycleRecoveryFixture(t, 2, true)
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!strings.Contains(err.Error(), "found 2 unresolved preparations") {
		t.Fatalf("ambiguous stale lifecycle recovery = %v", err)
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	for _, preparation := range fixture.stale {
		if _, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID); err != nil {
			t.Fatal(err)
		} else if found {
			t.Fatalf("ambiguous recovery finalized %s", preparation.ID)
		}
	}
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.currentFinalization.ID {
		t.Fatalf("ambiguous recovery moved current authority to %s", current)
	}
}

func TestRecordFailureClosesSelectedStalePreparation(t *testing.T) {
	t.Parallel()
	fixture := newStaleLifecycleRecoveryFixture(t, 2, true)
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, "not-an-id"); err == nil ||
		!strings.Contains(err.Error(), "not an exact evidence artifact ID") {
		t.Fatalf("malformed selector = %v", err)
	}
	unknown, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("unknown selected preparation"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, unknown.String()); err == nil ||
		!strings.Contains(err.Error(), "not outstanding recovery debt") {
		t.Fatalf("unknown selector = %v", err)
	}
	if _, err := recordSelectedUnbatchableFailure(
		fixture.repo, fixture.storePath, fixture.currentFinalization.Preparation.String(),
	); err == nil || !strings.Contains(err.Error(), "not outstanding recovery debt") {
		t.Fatalf("finalized selector = %v", err)
	}
	heartbeatBefore, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(gateHeartbeatFile)))
	if err != nil {
		t.Fatal(err)
	}
	for _, stale := range fixture.stale {
		got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, stale.ID.String())
		if err != nil || got != stale.ID {
			t.Fatalf("selected recovery = (%s, %v)", got, err)
		}
	}
	heartbeatAfter, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(gateHeartbeatFile)))
	if err != nil {
		t.Fatal(err)
	}
	if string(heartbeatAfter) != string(heartbeatBefore) {
		t.Fatal("selected recovery mutated the finalized lifecycle locator")
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	for _, stale := range fixture.stale {
		finalization, found, err := runrecord.GateFinalizationForPreparation(ctx, store, stale.ID)
		if err != nil {
			t.Fatal(err)
		}
		if !found || finalization.Outcome != runrecord.OutcomeCancelled {
			t.Fatalf("selected recovery left %s unresolved", stale.ID)
		}
	}
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.currentFinalization.ID {
		t.Fatalf("selected recovery moved current authority to %s", current)
	}
	if err := requireNoOutstandingGateLifecycle(ctx, store); err != nil {
		t.Fatalf("selected recovery left outstanding debt: %v", err)
	}
}

func TestRecordFailureValidatesNewerFinalizedAuthorityBeforeClosingStaleDebt(t *testing.T) {
	t.Parallel()
	fixture := newStaleLifecycleRecoveryFixture(t, 1, false)
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!strings.Contains(err.Error(), "typed gate result") {
		t.Fatalf("incomplete current authority recovery = %v", err)
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, fixture.stale[0].ID,
	); err != nil {
		t.Fatal(err)
	} else if found {
		t.Fatal("stale preparation was finalized behind incomplete current authority")
	}
}

func TestRecordFailureBootstrapsLegacyAliasToNewestTerminalAuthority(t *testing.T) {
	t.Parallel()
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	heartbeatBefore, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(gateHeartbeatFile)))
	if err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err != nil ||
		got != fixture.outstanding[0].ID {
		t.Fatalf("legacy lifecycle recovery = (%s, %v)", got, err)
	}
	heartbeatAfter, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(gateHeartbeatFile)))
	if err != nil {
		t.Fatal(err)
	}
	if string(heartbeatAfter) == string(heartbeatBefore) {
		t.Fatal("legacy lifecycle recovery left the older finalized heartbeat in place")
	}
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(fixture.repo, gateHeartbeatFile, &heartbeat); err != nil {
		t.Fatal(err)
	}
	if heartbeat.State != runrecord.HeartbeatFinalized ||
		heartbeat.Preparation != fixture.newestPreparation.ID ||
		heartbeat.TreeKey != fixture.newestPreparation.TreeKey ||
		heartbeat.Environment != fixture.newestPreparation.Environment {
		t.Fatalf("bootstrapped lifecycle heartbeat = %+v", heartbeat)
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.newestFinalization.ID {
		t.Fatalf("bootstrapped lifecycle authority = (%s, %t), want newest %s", current, found, fixture.newestFinalization.ID)
	}
	if current == fixture.heartbeatFinalization.ID {
		t.Fatal("legacy bootstrap rewound to the older heartbeat finalization")
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(ctx, store, fixture.outstanding[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	if !found || finalization.Outcome != runrecord.OutcomeCancelled || finalization.ID == current {
		t.Fatalf("legacy stale cancellation = (%+v, %t), current=%s", finalization, found, current)
	}
	if err := validateCompleteGateFinalization(ctx, store, fixture.outstanding[0], finalization); err != nil {
		t.Fatalf("legacy stale cancellation is incomplete: %v", err)
	}
	if err := requireNoOutstandingGateLifecycle(ctx, store); err != nil {
		t.Fatalf("legacy bootstrap left lifecycle debt: %v", err)
	}
}

func TestRecordFailureRetryRepairsHeartbeatAfterLegacyStoreCommit(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
		Preparation: fixture.outstanding[0].ID, TreeKey: fixture.outstanding[0].TreeKey,
		Environment: fixture.outstanding[0].Environment, PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(fixture.repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	previousHook := gateRecordFailureAfterStoreCommitHook
	t.Cleanup(func() { gateRecordFailureAfterStoreCommitHook = previousHook })
	injected := errors.New("injected post-store heartbeat failure")
	gateRecordFailureAfterStoreCommitHook = func(*overgodb.Store) error { return injected }
	if got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); got != fixture.outstanding[0].ID ||
		!errors.Is(err, injected) {
		t.Fatalf("interrupted legacy heartbeat publication = (%s, %v)", got, err)
	}
	gateRecordFailureAfterStoreCommitHook = nil
	if err := readJSON(fixture.repo, gateHeartbeatFile, &heartbeat); err != nil {
		t.Fatal(err)
	}
	if heartbeat.State != runrecord.HeartbeatRunning || heartbeat.Preparation != fixture.outstanding[0].ID {
		t.Fatalf("post-store failpoint changed stale heartbeat = %+v", heartbeat)
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	headBefore, sequenceBefore := store.Head()
	current, found, err := artifact.ResolveAlias(
		t.Context(), store, runrecord.GateLifecycleCurrentAlias,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.newestFinalization.ID {
		t.Fatalf("interrupted bootstrap authority = (%s, %t), want %s", current, found, fixture.newestFinalization.ID)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err != nil ||
		got != fixture.outstanding[0].ID {
		t.Fatalf("retry completed legacy heartbeat publication = (%s, %v)", got, err)
	}
	store, err = overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	if head, sequence := store.Head(); head != headBefore || sequence != sequenceBefore {
		store.Close()
		t.Fatalf("heartbeat-only retry moved store head from (%s, %d) to (%s, %d)", headBefore, sequenceBefore, head, sequence)
	}
	if current, found, err := artifact.ResolveAlias(
		t.Context(), store, runrecord.GateLifecycleCurrentAlias,
	); err != nil {
		store.Close()
		t.Fatal(err)
	} else if !found || current != fixture.newestFinalization.ID {
		store.Close()
		t.Fatalf("heartbeat-only retry changed current authority to (%s, %t)", current, found)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(fixture.repo, gateHeartbeatFile, &heartbeat); err != nil {
		t.Fatal(err)
	}
	if heartbeat.State != runrecord.HeartbeatFinalized ||
		heartbeat.Preparation != fixture.newestPreparation.ID ||
		heartbeat.TreeKey != fixture.newestPreparation.TreeKey ||
		heartbeat.Environment != fixture.newestPreparation.Environment {
		t.Fatalf("heartbeat-only retry locator = %+v", heartbeat)
	}
	if err := requireNoPendingGateState(fixture.repo, fixture.storePath); err != nil {
		t.Fatalf("heartbeat-only retry left pending gate state: %v", err)
	}
}

func TestRecordFailureBootstrapsLegacyAliasWhenHeartbeatIsAbsent(t *testing.T) {
	t.Parallel()
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	if err := os.Remove(filepath.Join(fixture.repo, filepath.FromSlash(gateHeartbeatFile))); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err != nil ||
		got != fixture.outstanding[0].ID {
		t.Fatalf("heartbeat-absent legacy recovery = (%s, %v)", got, err)
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	current, found, err := artifact.ResolveAlias(
		t.Context(), store, runrecord.GateLifecycleCurrentAlias,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.newestFinalization.ID {
		t.Fatalf("heartbeat-absent bootstrap authority = (%s, %t), want %s", current, found, fixture.newestFinalization.ID)
	}
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(fixture.repo, gateHeartbeatFile, &heartbeat); err != nil {
		t.Fatal(err)
	}
	if heartbeat.State != runrecord.HeartbeatFinalized ||
		heartbeat.Preparation != fixture.newestPreparation.ID ||
		heartbeat.TreeKey != fixture.newestPreparation.TreeKey ||
		heartbeat.Environment != fixture.newestPreparation.Environment {
		t.Fatalf("heartbeat-absent bootstrap locator = %+v", heartbeat)
	}
}

func TestRecordFailurePrefersCanonicalLegacyDebtToUncommittedRunningLocator(t *testing.T) {
	t.Parallel()
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	uncommitted, err := runrecord.NewGatePreparation(
		strings.Repeat("a", 64), fixture.environment.ID, time.Unix(625, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
		Preparation: uncommitted.ID, TreeKey: uncommitted.TreeKey, Environment: uncommitted.Environment,
		PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(fixture.repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err != nil ||
		got != fixture.outstanding[0].ID {
		t.Fatalf("canonical debt behind uncommitted locator = (%s, %v)", got, err)
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.newestFinalization.ID {
		t.Fatalf("canonical locator recovery authority = (%s, %t), want %s", current, found, fixture.newestFinalization.ID)
	}
	if finalization, found, err := runrecord.GateFinalizationForPreparation(
		ctx, store, fixture.outstanding[0].ID,
	); err != nil {
		t.Fatal(err)
	} else if !found || finalization.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("canonical locator recovery finalization = (%+v, %t)", finalization, found)
	}
	if err := readJSON(fixture.repo, gateHeartbeatFile, &heartbeat); err != nil {
		t.Fatal(err)
	}
	if heartbeat.State != runrecord.HeartbeatFinalized || heartbeat.Preparation != fixture.newestPreparation.ID {
		t.Fatalf("canonical locator recovery heartbeat = %+v", heartbeat)
	}
}

func TestRecordFailureRemovesUncommittedRunningLocatorWhenLegacyDebtIsZero(t *testing.T) {
	t.Parallel()
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	publishUnaliasedGateFinalization(
		t, store, fixture.outstanding[0], fixture.environment, "legacy-zero-debt",
	)
	headBefore, sequenceBefore := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	uncommitted, err := runrecord.NewGatePreparation(
		strings.Repeat("b", 64), fixture.environment.ID, time.Unix(630, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
		Preparation: uncommitted.ID, TreeKey: uncommitted.TreeKey, Environment: uncommitted.Environment,
		PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(fixture.repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err != nil || got != uncommitted.ID {
		t.Fatalf("zero-debt uncommitted locator cleanup = (%s, %v)", got, err)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, filepath.FromSlash(gateHeartbeatFile))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("zero-debt uncommitted locator remains: %v", err)
	}
	store, err = overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if head, sequence := store.Head(); head != headBefore || sequence != sequenceBefore {
		t.Fatalf("zero-debt locator cleanup moved store head from (%s, %d) to (%s, %d)", headBefore, sequenceBefore, head, sequence)
	}
	if current, found, err := artifact.ResolveAlias(
		t.Context(), store, runrecord.GateLifecycleCurrentAlias,
	); err != nil {
		t.Fatal(err)
	} else if found {
		t.Fatalf("zero-debt locator cleanup published lifecycle alias %s", current)
	}
}

func TestRecordFailureBootstrapsLegacyAliasToCancellationWithoutPriorTerminal(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "legacy-cancellation@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Legacy Cancellation Test")
	runGitFixture(t, repo, "commit", "--allow-empty", "-q", "-m", "baseline")
	environment := lifecycleTestEnvironment(t)
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	environmentBatch, err := artifact.NewDocumentBatch(
		"test/legacy-cancellation/environment", []artifact.Content{environmentContent}, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), environmentBatch); err != nil {
		t.Fatal(err)
	}
	preparation := publishLegacyGatePreparation(
		t, store, environment, strings.Repeat("8", 64), time.Unix(628, 0), "only-outstanding",
	)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
		Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
		PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(repo, storePath, ""); err != nil || got != preparation.ID {
		t.Fatalf("legacy cancellation bootstrap = (%s, %v)", got, err)
	}
	store, err = overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	finalization, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, preparation.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || finalization.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("legacy cancellation finalization = (%+v, %t)", finalization, found)
	}
	current, found, err := artifact.ResolveAlias(
		t.Context(), store, runrecord.GateLifecycleCurrentAlias,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != finalization.ID {
		t.Fatalf("legacy cancellation authority = (%s, %t), want %s", current, found, finalization.ID)
	}
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); err != nil {
		t.Fatal(err)
	}
	if heartbeat.State != runrecord.HeartbeatFinalized || heartbeat.Preparation != preparation.ID {
		t.Fatalf("legacy cancellation heartbeat = %+v", heartbeat)
	}
}

func TestRecordFailureRefusesLegacyFinalizedHeartbeatWithoutExactFinalization(t *testing.T) {
	t.Parallel()
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatFinalized,
		Preparation: fixture.outstanding[0].ID, TreeKey: fixture.outstanding[0].TreeKey,
		Environment: fixture.environment.ID, PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(fixture.repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!strings.Contains(err.Error(), "locator preparation lacks a finalization") {
		t.Fatalf("legacy heartbeat without finalization = %v", err)
	}
	requireLegacyRecoveryUnchanged(t, fixture)
}

func TestRecordFailureRefusesMalformedLegacyFinalizationHistory(t *testing.T) {
	t.Parallel()
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	malformedPreparation := publishLegacyGatePreparation(
		t, store, fixture.environment, strings.Repeat("6", 64), time.Unix(626, 0), "malformed",
	)
	result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("legacy untyped gate result"))
	if err != nil {
		t.Fatal(err)
	}
	malformed, err := runrecord.NewGateFinalization(
		malformedPreparation, "0123456789abcdef0123456789abcdef01234567", result, runrecord.OutcomeFailed,
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := malformed.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"test/legacy-lifecycle/malformed-finalization",
		[]artifact.Content{content}, malformed.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: result})
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!strings.Contains(err.Error(), "typed gate result") {
		t.Fatalf("malformed legacy lifecycle history = %v", err)
	}
	requireLegacyRecoveryUnchanged(t, fixture)
}

func TestRecordFailureRefusesAmbiguousLegacyOutstandingPreparations(t *testing.T) {
	t.Parallel()
	fixture := newLegacyLifecycleRecoveryFixture(t, 2)
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!strings.Contains(err.Error(), "found 2 unresolved preparations") {
		t.Fatalf("ambiguous legacy lifecycle recovery = %v", err)
	}
	requireLegacyRecoveryUnchanged(t, fixture)
}

func TestRecordFailureLegacyBootstrapUsesDocumentOrderForCoIntroducedFinalizations(t *testing.T) {
	t.Parallel()
	fixture, first, second := newCoIntroducedLegacyRecoveryFixture(t, false)
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	firstIntroduction, found, err := store.ArtifactIntroduction(t.Context(), first.ID)
	if err != nil || !found {
		t.Fatalf("first co-introduced finalization = (%+v, %t, %v)", firstIntroduction, found, err)
	}
	secondIntroduction, found, err := store.ArtifactIntroduction(t.Context(), second.ID)
	if err != nil || !found {
		t.Fatalf("second co-introduced finalization = (%+v, %t, %v)", secondIntroduction, found, err)
	}
	if firstIntroduction.Sequence != secondIntroduction.Sequence {
		t.Fatalf("co-introduced finalization sequences = %d and %d", firstIntroduction.Sequence, secondIntroduction.Sequence)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err != nil ||
		got != fixture.outstanding[0].ID {
		t.Fatalf("co-introduced legacy recovery = (%s, %v)", got, err)
	}
	store, err = overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	current, found, err := artifact.ResolveAlias(
		t.Context(), store, runrecord.GateLifecycleCurrentAlias,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.newestFinalization.ID {
		t.Fatalf("co-introduced bootstrap authority = (%s, %t), want %s", current, found, fixture.newestFinalization.ID)
	}
	if artifact.CompareID(first.ID, second.ID) < 0 && current != second.ID ||
		artifact.CompareID(first.ID, second.ID) > 0 && current != first.ID {
		t.Fatalf("co-introduced bootstrap ignored artifact-ID document order: first=%s second=%s current=%s", first.ID, second.ID, current)
	}
}

func TestRecordFailureLegacyBootstrapRefusesDuplicateFinalizationPerPreparation(t *testing.T) {
	t.Parallel()
	fixture, _, _ := newCoIntroducedLegacyRecoveryFixture(t, true)
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!strings.Contains(err.Error(), "has 2 canonical finalizations") {
		t.Fatalf("duplicate legacy finalization recovery = %v", err)
	}
	requireLegacyRecoveryUnchanged(t, fixture)
}

func TestRecordFailureLegacyBootstrapRefusesHeadMovementWithoutAlias(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	previousHook := gateRecordFailureBeforeStoreCommitHook
	t.Cleanup(func() { gateRecordFailureBeforeStoreCommitHook = previousHook })
	gateRecordFailureBeforeStoreCommitHook = func(store *overgodb.Store) {
		preparation := publishLegacyGatePreparation(
			t, store, fixture.environment, strings.Repeat("7", 64), time.Unix(627, 0), "head-race",
		)
		publishUnaliasedGateFinalization(t, store, preparation, fixture.environment, "legacy-head-race")
	}
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!errors.Is(err, artifact.ErrCommitPrecondition) {
		t.Fatalf("legacy lifecycle head race = %v", err)
	}
	requireLegacyRecoveryUnchanged(t, fixture)
}

func TestRecordFailureLegacyBootstrapRefusesAliasRace(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	fixture := newLegacyLifecycleRecoveryFixture(t, 1)
	previousHook := gateRecordFailureBeforeStoreCommitHook
	t.Cleanup(func() { gateRecordFailureBeforeStoreCommitHook = previousHook })
	gateRecordFailureBeforeStoreCommitHook = func(store *overgodb.Store) {
		_, err := store.Commit(t.Context(), artifact.Batch{
			Key: "test/legacy-lifecycle/alias-race",
			Aliases: []artifact.AliasBinding{{
				Name: runrecord.GateLifecycleCurrentAlias, Target: fixture.newestFinalization.ID,
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if _, err := recordSelectedUnbatchableFailure(fixture.repo, fixture.storePath, ""); err == nil ||
		!errors.Is(err, artifact.ErrCommitPrecondition) {
		t.Fatalf("legacy lifecycle alias race = %v", err)
	}
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	current, found, err := artifact.ResolveAlias(t.Context(), store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != fixture.newestFinalization.ID {
		t.Fatalf("racing alias authority = (%s, %t), want %s", current, found, fixture.newestFinalization.ID)
	}
	if _, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, fixture.outstanding[0].ID,
	); err != nil {
		t.Fatal(err)
	} else if found {
		t.Fatal("alias race allowed stale cancellation to publish")
	}
}

func TestSoleCurrentPreparationRejectsExistingUnaliasedFinalization(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	environment := lifecycleTestEnvironment(t)
	preparation, err := runrecord.NewGatePreparation(
		"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
		environment.ID,
		time.Unix(402, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparationCommit := publishInterruptedPreparation(t, repo, storePath, environment, preparation)
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := requireSoleCurrentGatePreparation(
		t.Context(), store, preparation, preparationCommit,
	); err != nil {
		t.Fatalf("fresh preparation authority = %v", err)
	}
	publishUnaliasedGateFinalization(t, store, preparation, environment, "existing-finalization")
	if err := requireSoleCurrentGatePreparation(
		t.Context(), store, preparation, preparationCommit,
	); err == nil || !strings.Contains(err.Error(), "already has a finalization") {
		t.Fatalf("pre-finalized preparation admitted = %v", err)
	}
}

func TestFinalRecordRefusesSecondFinalization(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	environment := lifecycleTestEnvironment(t)
	preparation, err := runrecord.NewGatePreparation(
		"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
		environment.ID,
		time.Unix(404, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparationCommit := publishInterruptedPreparation(t, repo, storePath, environment, preparation)
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	retained, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Close()
	publishUnaliasedGateFinalization(t, store, preparation, environment, "record-conflict")
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	g := gateContext{
		repo: repo, storePath: storePath, start: time.Now().Add(-time.Second),
		environment: environment, preparation: preparation, preparationCommit: preparationCommit,
		planHead: "0123456789abcdef0123456789abcdef01234567", planRef: "authority/record",
		store: retained,
		steps: []runrecord.GateStep{{
			Name: "scope", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepFailed, DurationNS: 1,
		}},
	}
	if err := g.record(runrecord.OutcomeFailed, "scope"); err == nil ||
		!strings.Contains(err.Error(), "lost preparation authority") {
		t.Fatalf("conflicting final record = %v", err)
	}
	store, err = overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	_, found, err := runrecord.GateFinalizationForPreparation(
		t.Context(), store, preparation.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("final record did not retain its sole finalization")
	}
}

func TestRecordFailureBindsExistingUnaliasedFinalization(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	environment := lifecycleTestEnvironment(t)
	preparation, err := runrecord.NewGatePreparation(
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
		environment.ID,
		time.Unix(405, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = publishInterruptedPreparation(t, repo, storePath, environment, preparation)
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	finalization := publishUnaliasedGateFinalization(t, store, preparation, environment, "repair-alias")
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: repo, environment: environment, preparation: preparation}
	if err := g.writeHeartbeat(runrecord.HeartbeatRunning); err != nil {
		t.Fatal(err)
	}
	if got, err := recordSelectedUnbatchableFailure(repo, storePath, ""); err != nil || got != preparation.ID {
		t.Fatalf("repair unaliased finalization = (%s, %v)", got, err)
	}
	store, err = overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	current, found, err := artifact.ResolveAlias(
		t.Context(), store, runrecord.GateLifecycleCurrentAlias,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !found || current != finalization.ID {
		t.Fatalf("repaired lifecycle alias = (%s, %t), want %s", current, found, finalization.ID)
	}
}

func TestFinalizedLifecycleAliasDoesNotHideForgedUnaliasedFinalization(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, environment := newCompleteLifecycleAliasFixture(t)
	preparation, err := runrecord.NewGatePreparation(
		"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
		environment.ID,
		time.Unix(403, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparationContent, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("untyped hidden gate result"))
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(
		preparation, "0123456789abcdef0123456789abcdef01234567", result, runrecord.OutcomeFailed,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"test/finalized-alias/forged-unaliased-finalization",
		[]artifact.Content{preparationContent, finalizationContent},
		append(preparation.Lineage(), finalization.Lineage()...),
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: result})
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if err := requireNoOutstandingGateLifecycle(ctx, store); err == nil || !strings.Contains(err.Error(), "typed gate result") {
		t.Fatalf("good current alias with forged later finalization = %v", err)
	}
}

func newLifecycleRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	return repo
}

type staleLifecycleRecoveryFixture struct {
	repo                string
	storePath           string
	stale               []runrecord.GateLifecycle
	currentFinalization runrecord.GateLifecycle
}

func newStaleLifecycleRecoveryFixture(
	t *testing.T,
	staleCount int,
	typedCurrent bool,
) staleLifecycleRecoveryFixture {
	t.Helper()
	if staleCount < 1 || staleCount > 2 {
		t.Fatalf("unsupported stale lifecycle fixture count %d", staleCount)
	}
	repo, storePath := newLifecycleRepo(t), "store"
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "stale-lifecycle@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Stale Lifecycle Test")
	runGitFixture(t, repo, "commit", "--allow-empty", "-q", "-m", "baseline")
	environment := lifecycleTestEnvironment(t)
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			_ = store.Close()
		}
	}()
	fixture := staleLifecycleRecoveryFixture{repo: repo, storePath: storePath}
	prepareContents := []artifact.Content{environmentContent}
	var prepareLineage []artifact.Lineage
	for index, key := range []string{
		strings.Repeat("1", 64), strings.Repeat("2", 64),
	}[:staleCount] {
		preparation, err := runrecord.NewGatePreparation(key, environment.ID, time.Unix(500+int64(index), 0))
		if err != nil {
			t.Fatal(err)
		}
		content, err := preparation.Content()
		if err != nil {
			t.Fatal(err)
		}
		fixture.stale = append(fixture.stale, preparation)
		prepareContents = append(prepareContents, content)
		prepareLineage = append(prepareLineage, preparation.Lineage()...)
	}
	prepareBatch, err := artifact.NewDocumentBatch(
		"test/stale-lifecycle/preparations", prepareContents, prepareLineage, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), prepareBatch); err != nil {
		t.Fatal(err)
	}
	currentPreparation, err := runrecord.NewGatePreparation(
		strings.Repeat("3", 64), environment.ID, time.Unix(510, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	currentContent, err := currentPreparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	currentBatch, err := artifact.NewDocumentBatch(
		"test/stale-lifecycle/current-preparation",
		[]artifact.Content{currentContent}, currentPreparation.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), currentBatch); err != nil {
		t.Fatal(err)
	}
	if typedCurrent {
		recipe, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("stale lifecycle current recipe"))
		if err != nil {
			t.Fatal(err)
		}
		record, err := runrecord.NewGateRecord(
			recipe, environment.ID, "0123456789abcdef0123456789abcdef01234567",
			runrecord.OutcomeSucceeded, "", 1,
			[]runrecord.GateStep{{
				Name: "verify", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1,
			}},
		)
		if err != nil {
			t.Fatal(err)
		}
		fixture.currentFinalization, err = runrecord.NewGateFinalization(
			currentPreparation, record.Result.CodeCommit, record.Result.ID, record.Result.Outcome,
		)
		if err != nil {
			t.Fatal(err)
		}
		finalizationContent, err := fixture.currentFinalization.Content()
		if err != nil {
			t.Fatal(err)
		}
		finalBatch, err := record.Batch("test/stale-lifecycle/current-finalization")
		if err != nil {
			t.Fatal(err)
		}
		finalBatch.Artifacts = append(finalBatch.Artifacts, artifact.Descriptor{ID: recipe})
		finalBatch.Contents = append(finalBatch.Contents, finalizationContent)
		finalBatch.Lineage = append(finalBatch.Lineage, fixture.currentFinalization.Lineage()...)
		finalBatch.Aliases = append(finalBatch.Aliases, artifact.AliasBinding{
			Name: runrecord.GateLifecycleCurrentAlias, Target: fixture.currentFinalization.ID,
		})
		if _, err := store.Commit(t.Context(), finalBatch); err != nil {
			t.Fatal(err)
		}
	} else {
		result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("untyped current gate result"))
		if err != nil {
			t.Fatal(err)
		}
		fixture.currentFinalization, err = runrecord.NewGateFinalization(
			currentPreparation, "0123456789abcdef0123456789abcdef01234567", result, runrecord.OutcomeFailed,
		)
		if err != nil {
			t.Fatal(err)
		}
		finalizationContent, err := fixture.currentFinalization.Content()
		if err != nil {
			t.Fatal(err)
		}
		finalBatch, err := artifact.NewDocumentBatch(
			"test/stale-lifecycle/incomplete-current-finalization",
			[]artifact.Content{finalizationContent}, fixture.currentFinalization.Lineage(),
			[]artifact.AliasBinding{{
				Name: runrecord.GateLifecycleCurrentAlias, Target: fixture.currentFinalization.ID,
			}},
		)
		if err != nil {
			t.Fatal(err)
		}
		finalBatch.Artifacts = append(finalBatch.Artifacts, artifact.Descriptor{ID: result})
		if _, err := store.Commit(t.Context(), finalBatch); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatFinalized,
		Preparation: currentPreparation.ID, TreeKey: currentPreparation.TreeKey, Environment: environment.ID,
		PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	return fixture
}

type legacyLifecycleRecoveryFixture struct {
	repo                  string
	storePath             string
	environment           runrecord.Environment
	outstanding           []runrecord.GateLifecycle
	heartbeatPreparation  runrecord.GateLifecycle
	heartbeatFinalization runrecord.GateLifecycle
	newestPreparation     runrecord.GateLifecycle
	newestFinalization    runrecord.GateLifecycle
}

func newLegacyLifecycleRecoveryFixture(t *testing.T, outstandingCount int) legacyLifecycleRecoveryFixture {
	t.Helper()
	if outstandingCount < 1 || outstandingCount > 2 {
		t.Fatalf("unsupported legacy outstanding count %d", outstandingCount)
	}
	repo, storePath := newLifecycleRepo(t), "store"
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "legacy-lifecycle@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Legacy Lifecycle Test")
	runGitFixture(t, repo, "commit", "--allow-empty", "-q", "-m", "baseline")
	fixture := legacyLifecycleRecoveryFixture{
		repo: repo, storePath: storePath, environment: lifecycleTestEnvironment(t),
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			_ = store.Close()
		}
	}()
	environmentContent, err := fixture.environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	environmentBatch, err := artifact.NewDocumentBatch(
		"test/legacy-lifecycle/environment", []artifact.Content{environmentContent}, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), environmentBatch); err != nil {
		t.Fatal(err)
	}
	for index, treeKey := range []string{
		strings.Repeat("1", 64), strings.Repeat("2", 64),
	}[:outstandingCount] {
		fixture.outstanding = append(fixture.outstanding, publishLegacyGatePreparation(
			t, store, fixture.environment, treeKey, time.Unix(610+int64(index), 0),
			"outstanding-"+treeKey[:1],
		))
	}
	fixture.heartbeatPreparation = publishLegacyGatePreparation(
		t, store, fixture.environment, strings.Repeat("3", 64), time.Unix(620, 0), "heartbeat",
	)
	fixture.heartbeatFinalization = publishUnaliasedGateFinalization(
		t, store, fixture.heartbeatPreparation, fixture.environment, "legacy-heartbeat",
	)
	fixture.newestPreparation = publishLegacyGatePreparation(
		t, store, fixture.environment, strings.Repeat("4", 64), time.Unix(621, 0), "newest",
	)
	fixture.newestFinalization = publishUnaliasedGateFinalization(
		t, store, fixture.newestPreparation, fixture.environment, "legacy-newest",
	)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatFinalized,
		Preparation: fixture.heartbeatPreparation.ID, TreeKey: fixture.heartbeatPreparation.TreeKey,
		Environment: fixture.environment.ID, PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func newCoIntroducedLegacyRecoveryFixture(
	t *testing.T,
	duplicatePreparation bool,
) (legacyLifecycleRecoveryFixture, runrecord.GateLifecycle, runrecord.GateLifecycle) {
	t.Helper()
	repo, storePath := newLifecycleRepo(t), "store"
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "legacy-cointroduced@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Legacy Cointroduced Test")
	runGitFixture(t, repo, "commit", "--allow-empty", "-q", "-m", "baseline")
	fixture := legacyLifecycleRecoveryFixture{
		repo: repo, storePath: storePath, environment: lifecycleTestEnvironment(t),
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			_ = store.Close()
		}
	}()
	environmentContent, err := fixture.environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	environmentBatch, err := artifact.NewDocumentBatch(
		"test/legacy-cointroduced/environment", []artifact.Content{environmentContent}, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), environmentBatch); err != nil {
		t.Fatal(err)
	}
	fixture.outstanding = []runrecord.GateLifecycle{publishLegacyGatePreparation(
		t, store, fixture.environment, strings.Repeat("e", 64), time.Unix(633, 0), "cointroduced-outstanding",
	)}
	firstPreparation := publishLegacyGatePreparation(
		t, store, fixture.environment, strings.Repeat("f", 64), time.Unix(634, 0), "cointroduced-first",
	)
	secondPreparation := firstPreparation
	if !duplicatePreparation {
		secondPreparation = publishLegacyGatePreparation(
			t, store, fixture.environment, strings.Repeat("0", 64), time.Unix(635, 0), "cointroduced-second",
		)
	}
	firstBatch, firstFinalization := unaliasedGateFinalizationBatch(
		t, firstPreparation, fixture.environment, "legacy-cointroduced-first",
	)
	secondBatch, secondFinalization := unaliasedGateFinalizationBatch(
		t, secondPreparation, fixture.environment, "legacy-cointroduced-second",
	)
	combined := firstBatch
	combined.Key = "test/legacy-cointroduced/finalizations"
	combined.Artifacts = append(combined.Artifacts, secondBatch.Artifacts...)
	combined.Contents = append(combined.Contents, secondBatch.Contents...)
	combined.Manifests = append(combined.Manifests, secondBatch.Manifests...)
	combined.Lineage = append(combined.Lineage, secondBatch.Lineage...)
	combined.Causality = append(combined.Causality, secondBatch.Causality...)
	combined.Aliases = append(combined.Aliases, secondBatch.Aliases...)
	combined.Locations = append(combined.Locations, secondBatch.Locations...)
	if _, err := store.Commit(t.Context(), combined); err != nil {
		t.Fatal(err)
	}
	fixture.heartbeatPreparation = firstPreparation
	fixture.heartbeatFinalization = firstFinalization
	fixture.newestPreparation = secondPreparation
	fixture.newestFinalization = secondFinalization
	if artifact.CompareID(firstFinalization.ID, secondFinalization.ID) > 0 {
		fixture.heartbeatPreparation = secondPreparation
		fixture.heartbeatFinalization = secondFinalization
		fixture.newestPreparation = firstPreparation
		fixture.newestFinalization = firstFinalization
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatFinalized,
		Preparation: fixture.heartbeatPreparation.ID, TreeKey: fixture.heartbeatPreparation.TreeKey,
		Environment: fixture.environment.ID, PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	if err := writeJSON(repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		t.Fatal(err)
	}
	return fixture, firstFinalization, secondFinalization
}

func publishLegacyGatePreparation(
	t *testing.T,
	store *overgodb.Store,
	environment runrecord.Environment,
	treeKey string,
	started time.Time,
	key string,
) runrecord.GateLifecycle {
	t.Helper()
	preparation, err := runrecord.NewGatePreparation(treeKey, environment.ID, started)
	if err != nil {
		t.Fatal(err)
	}
	content, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"test/legacy-lifecycle/preparation/"+key,
		[]artifact.Content{content}, preparation.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return preparation
}

func requireLegacyRecoveryUnchanged(t *testing.T, fixture legacyLifecycleRecoveryFixture) {
	t.Helper()
	store, err := overgodb.Open(filepath.Join(fixture.repo, fixture.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias); err != nil {
		t.Fatal(err)
	} else if found {
		t.Fatalf("refused legacy recovery published lifecycle alias %s", current)
	}
	for _, preparation := range fixture.outstanding {
		if _, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID); err != nil {
			t.Fatal(err)
		} else if found {
			t.Fatalf("refused legacy recovery finalized %s", preparation.ID)
		}
	}
}

func lifecycleTestEnvironment(t *testing.T) runrecord.Environment {
	t.Helper()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	return environment
}

func publishUnaliasedGateFinalization(
	t *testing.T,
	store *overgodb.Store,
	preparation runrecord.GateLifecycle,
	environment runrecord.Environment,
	key string,
) runrecord.GateLifecycle {
	t.Helper()
	batch, finalization := unaliasedGateFinalizationBatch(t, preparation, environment, key)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return finalization
}

func unaliasedGateFinalizationBatch(
	t *testing.T,
	preparation runrecord.GateLifecycle,
	environment runrecord.Environment,
	key string,
) (artifact.Batch, runrecord.GateLifecycle) {
	t.Helper()
	recipe, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("lifecycle/"+key))
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewGateRecord(
		recipe,
		environment.ID,
		"0123456789abcdef0123456789abcdef01234567",
		runrecord.OutcomeCancelled,
		"",
		1,
		[]runrecord.GateStep{{
			Name: "recovery", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepCancelled, DurationNS: 1,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(
		preparation, record.Result.CodeCommit, record.Result.ID, record.Result.Outcome,
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch("test/lifecycle/" + key)
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipe})
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, finalization.Lineage()...)
	return batch, finalization
}

func lifecycleDebtBatch(
	t *testing.T,
	preparation runrecord.GateLifecycle,
	environment runrecord.Environment,
	key string,
) (artifact.Batch, runrecord.GateLifecycle) {
	t.Helper()
	recipe, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("lifecycle/debt/"+key))
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewGateRecord(
		recipe,
		environment.ID,
		"0123456789abcdef0123456789abcdef01234567",
		runrecord.OutcomeFailed,
		"record",
		1,
		[]runrecord.GateStep{{
			Name: "record", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepFailed, DurationNS: 1,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(
		preparation, record.Result.CodeCommit, record.Result.ID, record.Result.Outcome,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch("gate/final/" + preparation.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipe})
	batch.Contents = append(batch.Contents, environmentContent, finalizationContent)
	batch.Lineage = append(batch.Lineage, finalization.Lineage()...)
	appendGateFinalizationAlias(&batch, preparation.ID, finalization.ID)
	return batch, finalization
}

func newCompleteLifecycleAliasFixture(t *testing.T) (*overgodb.Store, runrecord.Environment) {
	t.Helper()
	ctx := t.Context()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		environment.ID,
		time.Unix(400, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	preparationContent, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	recipe, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("lifecycle recipe"))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close lifecycle fixture store: %v", err)
		}
	})
	prepareBatch, err := artifact.NewDocumentBatch(
		"test/finalized-alias/preparation",
		[]artifact.Content{environmentContent, preparationContent},
		preparation.Lineage(),
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	prepareBatch.Artifacts = append(prepareBatch.Artifacts, artifact.Descriptor{ID: recipe})
	if _, err := store.Commit(ctx, prepareBatch); err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewGateRecord(
		recipe,
		environment.ID,
		"0123456789abcdef0123456789abcdef01234567",
		runrecord.OutcomeSucceeded,
		"",
		1,
		[]runrecord.GateStep{{
			Name: "verify", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(
		preparation, record.Result.CodeCommit, record.Result.ID, record.Result.Outcome,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	finalBatch, err := record.Batch("test/finalized-alias/result")
	if err != nil {
		t.Fatal(err)
	}
	finalBatch.Contents = append(finalBatch.Contents, finalizationContent)
	finalBatch.Lineage = append(finalBatch.Lineage, finalization.Lineage()...)
	finalBatch.Aliases = append(finalBatch.Aliases, artifact.AliasBinding{
		Name: runrecord.GateLifecycleCurrentAlias, Target: finalization.ID,
	})
	if _, err := store.Commit(ctx, finalBatch); err != nil {
		t.Fatal(err)
	}
	return store, environment
}

func TestTerminalLifecycleAliasRequiresTypedGateResult(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	ctx := t.Context()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(
		"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
		environment.ID,
		time.Unix(402, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	preparationContent, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("untyped gate result"))
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(
		preparation, "0123456789abcdef0123456789abcdef01234567", result, runrecord.OutcomeFailed,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch, err := artifact.NewDocumentBatch(
		"test/incomplete-finalized-alias",
		[]artifact.Content{environmentContent, preparationContent, finalizationContent},
		append(preparation.Lineage(), finalization.Lineage()...),
		[]artifact.AliasBinding{{Name: runrecord.GateLifecycleCurrentAlias, Target: finalization.ID}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: result})
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if err := requireNoOutstandingGateLifecycle(ctx, store); err == nil || !strings.Contains(err.Error(), "typed gate result") {
		t.Fatalf("untyped terminal alias = %v", err)
	}
}

func TestTerminalLifecycleRejectsGateResultPublishedLater(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newTemporalLifecycleFixture(t, true)
	resultContent, err := fixture.record.Result.Content()
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := fixture.finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	finalBatch, err := artifact.NewDocumentBatch(
		"test/temporal-finalization-before-result",
		[]artifact.Content{finalizationContent},
		fixture.finalization.Lineage(),
		[]artifact.AliasBinding{{Name: runrecord.GateLifecycleCurrentAlias, Target: fixture.finalization.ID}},
	)
	if err != nil {
		t.Fatal(err)
	}
	finalBatch.Artifacts = append(finalBatch.Artifacts, resultContent.Descriptor)
	if _, err := fixture.store.Commit(ctx, finalBatch); err != nil {
		t.Fatal(err)
	}
	resultBatch, err := artifact.NewDocumentBatch(
		"test/temporal-late-result",
		[]artifact.Content{resultContent},
		fixture.record.Result.Lineage(),
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, resultBatch); err != nil {
		t.Fatal(err)
	}
	if err := requireNoOutstandingGateLifecycle(ctx, fixture.store); err == nil ||
		!strings.Contains(err.Error(), "not published atomically") {
		t.Fatalf("late typed result authority = %v", err)
	}
}

func TestTerminalLifecycleRejectsEnvironmentPublishedLater(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newTemporalLifecycleFixture(t, false)
	resultContent, err := fixture.record.Result.Content()
	if err != nil {
		t.Fatal(err)
	}
	finalizationContent, err := fixture.finalization.Content()
	if err != nil {
		t.Fatal(err)
	}
	finalBatch, err := artifact.NewDocumentBatch(
		"test/temporal-finalization-before-environment",
		[]artifact.Content{resultContent, finalizationContent},
		append(fixture.record.Result.Lineage(), fixture.finalization.Lineage()...),
		[]artifact.AliasBinding{{Name: runrecord.GateLifecycleCurrentAlias, Target: fixture.finalization.ID}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, finalBatch); err != nil {
		t.Fatal(err)
	}
	environmentContent, err := fixture.environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	environmentBatch, err := artifact.NewDocumentBatch(
		"test/temporal-late-environment",
		[]artifact.Content{environmentContent},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, environmentBatch); err != nil {
		t.Fatal(err)
	}
	if err := requireNoOutstandingGateLifecycle(ctx, fixture.store); err == nil ||
		!strings.Contains(err.Error(), "environment was published after") {
		t.Fatalf("late environment authority = %v", err)
	}
}

type temporalLifecycleFixture struct {
	store        *overgodb.Store
	environment  runrecord.Environment
	preparation  runrecord.GateLifecycle
	record       runrecord.GateRecord
	finalization runrecord.GateLifecycle
}

func newTemporalLifecycleFixture(t *testing.T, publishEnvironment bool) temporalLifecycleFixture {
	t.Helper()
	ctx := t.Context()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "none", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := runrecord.NewGatePreparation(
		"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
		environment.ID,
		time.Unix(404, 0),
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
	recipe, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("temporal lifecycle recipe"))
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewGateRecord(
		recipe,
		environment.ID,
		"0123456789abcdef0123456789abcdef01234567",
		runrecord.OutcomeSucceeded,
		"",
		1,
		[]runrecord.GateStep{{
			Name: "verify", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	finalization, err := runrecord.NewGateFinalization(
		preparation, record.Result.CodeCommit, record.Result.ID, record.Result.Outcome,
	)
	if err != nil {
		t.Fatal(err)
	}
	contents := []artifact.Content{preparationContent}
	if publishEnvironment {
		contents = append([]artifact.Content{environmentContent}, contents...)
	}
	prepareBatch, err := artifact.NewDocumentBatch(
		"test/temporal-preparation",
		contents,
		preparation.Lineage(),
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	prepareBatch.Artifacts = append(prepareBatch.Artifacts, artifact.Descriptor{ID: recipe})
	if !publishEnvironment {
		prepareBatch.Artifacts = append(prepareBatch.Artifacts, environmentContent.Descriptor)
	}
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close temporal lifecycle store: %v", err)
		}
	})
	if _, err := store.Commit(ctx, prepareBatch); err != nil {
		t.Fatal(err)
	}
	return temporalLifecycleFixture{
		store: store, environment: environment, preparation: preparation, record: record, finalization: finalization,
	}
}
