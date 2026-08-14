package main

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

func TestGateDebtReconciliation(t *testing.T) {
	repo, storePath := t.TempDir(), "store"
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
		"test/prepared", []artifact.Content{environmentContent, preparationContent}, prepared.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), prepareBatch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	resultID, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("result"))
	finalized, err := runrecord.NewGateFinalization(
		prepared, "0123456789abcdef0123456789abcdef01234567", resultID, runrecord.OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizedContent, _ := finalized.Content()
	batch := artifact.Batch{
		Key: "gate/final/" + prepared.ID.String(), Artifacts: []artifact.Descriptor{{ID: resultID}},
		Contents: []artifact.Content{finalizedContent}, Lineage: finalized.Lineage(),
	}
	g := gateContext{repo: repo, preparation: prepared}
	if err := g.oweRecord(batch, errors.New("injected RepoDB outage")); err == nil {
		t.Fatal("oweRecord hid the triggering failure")
	}
	if got, err := reconcileGateDebt(repo, storePath); err != nil || got != prepared.ID {
		t.Fatalf("reconcile = (%s, %v)", got, err)
	}
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("debt payload remains after reconciliation: %v", err)
	}
}

func TestGateHeartbeat(t *testing.T) {
	preparation, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("preparation"))
	environment, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("environment"))
	g := gateContext{
		repo: t.TempDir(), preparation: runrecord.GateLifecycle{
			ID: preparation, TreeKey: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		}, environment: runrecord.Environment{ID: environment},
	}
	if err := g.writeHeartbeat(runrecord.HeartbeatRunning); err != nil {
		t.Fatal(err)
	}
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(g.repo, gateHeartbeatFile, &heartbeat); err != nil {
		t.Fatal(err)
	}
	if heartbeat.Preparation != preparation || heartbeat.Environment != environment || heartbeat.PID != os.Getpid() {
		t.Fatalf("heartbeat omitted gate facts: %+v", heartbeat)
	}
}
