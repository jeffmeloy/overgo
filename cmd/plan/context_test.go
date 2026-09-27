package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestAutomationContextSnapshotEncoding(t *testing.T) {
	t.Parallel()
	document := plan.Plan{Items: []plan.Item{{
		ID: "automation", Title: "Automation", Status: "open",
		Steps: []plan.Step{{ID: "context", Title: "Context", Status: "open", Verify: "go test ./..."}},
	}}}
	worktree, head := mustTestRepositoryBinding(t, document)
	facts := plan.ContextFacts{
		Head: head, Branch: "codex/automation",
		Worktree: worktree, Role: "sqa",
		Workflow: plan.WorkflowContext{Phase: "sqa", Source: "git:HEAD+overgodb:overgodb-store"},
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

func TestReviewPriority(t *testing.T) {
	t.Parallel()
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
	evaluator, err := runrecord.NewReviewEvaluator("gate", testutil.ArtifactID(t, artifact.KindRecipe, "definition"), "0123456789abcdef0123456789abcdef01234567")
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := runrecord.NewReviewCandidate(runrecord.ReviewCandidate{
		BaseCommit: "0123456789abcdef0123456789abcdef01234567", CodeCommit: target,
		Developer: developer.ID, Worktree: developerTree.ID, Evaluator: evaluator.ID,
		GateResult: testutil.ArtifactID(t, artifact.KindEvidence, "result"), GateRun: testutil.ArtifactID(t, artifact.KindEvidence, "run"),
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
	store, err := overgodb.Open(filepath.Join(worktree, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.CommitAs(t.Context(), gateDoor, batch); err != nil {
		t.Fatal(err)
	}
	workflow := authoritativeContextEvidence(store, target)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if workflow.Phase != "sqa" || workflow.CandidateID != candidate.ID.String() || workflow.VerdictID != "" {
		t.Fatalf("review priority = %+v", workflow)
	}
}
