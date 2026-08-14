package gatecontrol

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

func TestGateLifecycleStore(t *testing.T) {
	if _, err := New(t.TempDir(), filepath.Join("..", "outside")); err == nil {
		t.Fatal("RepoDB path escaped the gate-control root")
	}
	store, prepared, batch := lifecycleFixture(t)
	exists, err := store.DebtExists()
	if err != nil || exists {
		t.Fatalf("initial debt = (%v, %v)", exists, err)
	}
	if err := store.PersistDebt(prepared.ID, batch); err != nil {
		t.Fatal(err)
	}
	exists, err = store.DebtExists()
	if err != nil || !exists {
		t.Fatalf("persisted debt = (%v, %v)", exists, err)
	}
}

func TestGateHeartbeatStore(t *testing.T) {
	root := t.TempDir()
	store, err := New(root, "store")
	if err != nil {
		t.Fatal(err)
	}
	status, err := store.Watchdog(time.Unix(100, 0), time.Second)
	if err != nil || status.State != runrecord.HeartbeatAbsent || status.Heartbeat != nil {
		t.Fatalf("absent watchdog = (%+v, %v)", status, err)
	}
	preparation, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("preparation"))
	environment, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("environment"))
	heartbeat := runrecord.GateHeartbeat{
		Version: runrecord.GateHeartbeatVersion, State: runrecord.HeartbeatRunning,
		Preparation: preparation, Environment: environment,
		TreeKey: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", PID: 1,
	}
	stop, err := store.StartHeartbeat(heartbeat, time.Hour, func() time.Time { return time.Unix(100, 0) })
	if err != nil {
		t.Fatal(err)
	}
	stop()
	status, err = store.Watchdog(time.Unix(102, 0), time.Second)
	if err != nil || status.State != runrecord.HeartbeatStale || status.Heartbeat == nil {
		t.Fatalf("stale watchdog = (%+v, %v)", status, err)
	}
}

func TestGateDebtReconciliation(t *testing.T) {
	store, prepared, batch := lifecycleFixture(t)
	if err := store.PersistDebt(prepared.ID, batch); err != nil {
		t.Fatal(err)
	}
	got, err := store.Reconcile(context.Background())
	if err != nil || got != prepared.ID {
		t.Fatalf("reconcile = (%s, %v), want %s", got, err, prepared.ID)
	}
	if _, err := os.Stat(filepath.Join(store.root, filepath.FromSlash(DebtFile))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("debt payload remains: %v", err)
	}
}

func lifecycleFixture(t *testing.T) (Store, runrecord.GateLifecycle, artifact.Batch) {
	t.Helper()
	root := t.TempDir()
	store, err := New(root, "store")
	if err != nil {
		t.Fatal(err)
	}
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
	environmentContent, _ := environment.Content()
	preparedContent, _ := prepared.Content()
	prepareBatch, err := artifact.NewDocumentBatch(
		"test/prepared", []artifact.Content{environmentContent, preparedContent}, prepared.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	repository, err := repodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := repository.Commit(context.Background(), prepareBatch); err != nil {
		t.Fatal(err)
	}
	if err := repository.Close(); err != nil {
		t.Fatal(err)
	}
	result, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("result"))
	finalized, err := runrecord.NewGateFinalization(
		prepared, "0123456789abcdef0123456789abcdef01234567", result, runrecord.OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizedContent, _ := finalized.Content()
	batch := artifact.Batch{
		Key: "gate/final/" + prepared.ID.String(), Artifacts: []artifact.Descriptor{{ID: result}},
		Contents: []artifact.Content{finalizedContent}, Lineage: finalized.Lineage(),
	}
	return store, prepared, batch
}
