package plan

import "testing"

func TestAutomationContextHasOneCurrentTask(t *testing.T) {
	document := Plan{Items: []Item{
		{ID: "current", Title: "Current item", Status: "open", Steps: []Step{
			{ID: "first", Title: "First step", Status: "open", Verify: "go test ./..."},
			{ID: "second", Title: "Second step", Status: "open", Verify: "go test ./..."},
		}},
		{ID: "later", Status: "open", Steps: []Step{{ID: "work", Status: "open", Verify: "go test ./..."}}},
	}}
	facts := ContextFacts{
		Head: "0123456789abcdef0123456789abcdef01234567", Branch: "codex/automation",
		Worktree: `C:\repo`, Role: "developer",
		Dirty:        []DirtyPath{{Path: `z\file.go`, WorktreeStatus: "M"}, {Path: "a/file.go", IndexStatus: "A"}},
		EvidenceDebt: EvidenceDebt{State: "none_observed", Source: "overgodb:overgodb-store"},
		Workflow:     WorkflowContext{Phase: "implementation", Source: "git:HEAD+overgodb:overgodb-store"},
	}
	ctx, err := BuildAutomationContext(
		document, facts, testCompletionAuthorityAt(t, document, facts.Worktree, facts.Head),
	)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.SchemaVersion != AutomationContextVersion || ctx.PlanState != "active" || ctx.CurrentTask == nil {
		t.Fatalf("context header/current = %+v", ctx)
	}
	if ctx.CurrentTask.ItemID != "current" || ctx.CurrentTask.StepID != "first" || ctx.CurrentTask.Verify != "go test ./..." {
		t.Fatalf("current task = %+v", ctx.CurrentTask)
	}
	if len(ctx.Dirty) != 2 || ctx.Dirty[0].Path != "a/file.go" || ctx.Dirty[1].Path != "z/file.go" {
		t.Fatalf("normalized dirty paths = %+v", ctx.Dirty)
	}
}

func TestReviewPriority(t *testing.T) {
	document := Plan{Items: []Item{{ID: "one", Title: "One", Status: "open", Steps: []Step{{ID: "do", Title: "Do", Status: "open", Verify: "go test ./..."}}}}}
	facts := ContextFacts{
		Head: "0123456789abcdef0123456789abcdef01234567", Branch: "codex/automation", Worktree: "C:/repo",
		EvidenceDebt: EvidenceDebt{State: "none_observed", Source: "overgodb:overgodb-store"},
		Workflow:     WorkflowContext{Phase: "sqa", Source: "git:HEAD+overgodb:overgodb-store", CandidateID: "evidence:fixture"},
	}
	ctx, err := BuildAutomationContext(
		document, facts, testCompletionAuthorityAt(t, document, facts.Worktree, facts.Head),
	)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.Workflow.Phase != "sqa" || ctx.Workflow.CandidateID != "evidence:fixture" || ctx.CurrentTask == nil || ctx.CurrentTask.ItemID != "one" {
		t.Fatalf("review priority context = %+v", ctx)
	}
	facts.Workflow.Phase = "review"
	if _, err := BuildAutomationContext(
		document, facts, testCompletionAuthorityAt(t, document, facts.Worktree, facts.Head),
	); err == nil {
		t.Fatal("invalid workflow phase accepted")
	}
}

func TestAutomationContextRefusesUnresolvedOrCrossPlanAuthority(t *testing.T) {
	document := Plan{Items: []Item{{
		ID: "one", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	facts := ContextFacts{
		Head: "0123456789abcdef0123456789abcdef01234567", Branch: "codex/automation", Worktree: "C:/repo",
		EvidenceDebt: EvidenceDebt{State: "none_observed", Source: "overgodb:overgodb-store"},
		Workflow:     WorkflowContext{Phase: "implementation", Source: "git:HEAD+overgodb:overgodb-store"},
	}
	if _, err := BuildAutomationContext(document, facts, CompletionAuthority{}); err == nil {
		t.Fatal("zero completion authority accepted")
	}
	other := Plan{Items: []Item{{
		ID: "other", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	if _, err := BuildAutomationContext(
		document, facts, testCompletionAuthorityAt(t, other, facts.Worktree, facts.Head),
	); err == nil {
		t.Fatal("cross-plan completion authority accepted")
	}
	authority := testCompletionAuthorityAt(t, document, facts.Worktree, facts.Head)
	stale := facts
	stale.Head = "89abcdef0123456789abcdef0123456789abcdef"
	if _, err := BuildAutomationContext(document, stale, authority); err == nil {
		t.Fatal("stale-revision completion authority accepted")
	}
	foreign := facts
	foreign.Worktree = facts.Worktree + "-other"
	if _, err := BuildAutomationContext(document, foreign, authority); err == nil {
		t.Fatal("cross-repository completion authority accepted")
	}
}
