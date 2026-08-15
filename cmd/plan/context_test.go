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
		Workflow:     plan.WorkflowContext{Phase: "sqa", Source: "git:HEAD+repodb:repodb-store"},
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
	if decoded.Role != "sqa" || decoded.Workflow.Phase != "sqa" || decoded.CurrentTask == nil || decoded.CurrentTask.ItemID != "automation" || decoded.CurrentTask.StepID != "context" {
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
	debt, _ := authoritativeContextEvidence(worktree, "0123456789abcdef0123456789abcdef01234567")
	if debt.State != "present" || debt.Source != "repodb:repodb-store" || debt.ResultID != prepared.ID.String() {
		t.Fatalf("authoritative debt = %+v", debt)
	}
}

func TestReviewPriority(t *testing.T) {
	const target = "89abcdef0123456789abcdef0123456789abcdef"
	worktree := t.TempDir()
	developer, err := runrecord.NewReviewActor("local:developer", runrecord.ReviewDeveloper)
	if err != nil {
		t.Fatal(err)
	}
	developerTree, err := runrecord.NewReviewWorktree("C:/repo/dev", "codex/dev", target, true)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := runrecord.NewReviewEvaluator("gate", contextArtifactID(t, artifact.KindRecipe, "definition"), "0123456789abcdef0123456789abcdef01234567")
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := runrecord.NewReviewCandidate(runrecord.ReviewCandidate{
		BaseCommit: "0123456789abcdef0123456789abcdef01234567", CodeCommit: target,
		Developer: developer.ID, Worktree: developerTree.ID, Evaluator: evaluator.ID,
		GateResult: contextArtifactID(t, artifact.KindEvidence, "result"), GateRun: contextArtifactID(t, artifact.KindEvidence, "run"),
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := candidate.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("test/review-priority", []artifact.Content{content}, nil, nil)
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
	_, workflow := authoritativeContextEvidence(worktree, target)
	if workflow.Phase != "sqa" || workflow.CandidateID != candidate.ID.String() || workflow.VerdictID != "" {
		t.Fatalf("review priority = %+v", workflow)
	}
}

func contextArtifactID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
