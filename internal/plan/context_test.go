package plan

import "testing"

func TestAutomationContextHasOneCurrentTask(t *testing.T) {
	document := Plan{Items: []Item{
		{ID: "current", Title: "Current item", Status: "open", Steps: []Step{
			{ID: "first", Title: "First step", Status: "open", Verify: "go test ./..."},
			{ID: "second", Title: "Second step", Status: "open"},
		}},
		{ID: "later", Status: "open", Steps: []Step{{ID: "work", Status: "open"}}},
	}}
	ctx, err := BuildAutomationContext(document, ContextFacts{
		Head: "0123456789abcdef0123456789abcdef01234567", Branch: "codex/automation",
		Worktree: `C:\repo`, Role: "developer",
		Dirty:        []DirtyPath{{Path: `z\file.go`, WorktreeStatus: "M"}, {Path: "a/file.go", IndexStatus: "A"}},
		EvidenceDebt: EvidenceDebt{State: "none_observed", Source: "repodb:repodb-store"},
	})
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
