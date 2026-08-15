package main

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/plan"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

func TestAutomationContextSnapshotEncoding(t *testing.T) {
	document := plan.Plan{Items: []plan.Item{{
		ID: "automation", Title: "Automation", Status: "open",
		Steps: []plan.Step{{ID: "context", Title: "Context", Status: "open", Verify: "go test ./..."}},
	}}}
	facts := plan.ContextFacts{
		Head: "0123456789abcdef0123456789abcdef01234567", Branch: "codex/automation",
		Worktree: "C:/repo", Role: "sqa",
		EvidenceDebt: plan.EvidenceDebt{State: "possible", Source: "bin/gate_status.json", Reason: "fixture"},
	}
	context, err := plan.BuildAutomationContext(document, facts)
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := json.NewEncoder(&output).Encode(context); err != nil {
		t.Fatal(err)
	}
	var decoded plan.AutomationContext
	if err := json.Unmarshal(output.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Role != "sqa" || decoded.CurrentTask == nil || decoded.CurrentTask.ItemID != "automation" || decoded.CurrentTask.StepID != "context" {
		t.Fatalf("encoded context = %+v", decoded)
	}
}

func TestGateDebtAutomationContext(t *testing.T) {
	worktree := t.TempDir()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "test", OS: "test", Arch: "test", Device: "host",
		Backend: "go", Driver: "cgo=0", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := runrecord.NewGatePreparation(
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		environment.ID, time.Unix(100, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, _ := environment.Content()
	preparedContent, _ := prepared.Content()
	batch, err := artifact.NewDocumentBatch(
		"test/context-debt", []artifact.Content{environmentContent, preparedContent}, prepared.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(filepath.Join(worktree, "repodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	debt := authoritativeEvidenceDebt(worktree)
	if debt.State != "present" || debt.Source != "repodb:repodb-store" || debt.ResultID != prepared.ID.String() {
		t.Fatalf("authoritative debt = %+v", debt)
	}
}
