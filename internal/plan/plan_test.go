package plan

import (
	"reflect"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

func TestAdvancePreservesHistoricalSchemaAndInput(t *testing.T) {
	t.Parallel()
	document := Plan{Items: []Item{{ID: "item", Steps: []Step{
		{ID: ".", Verify: "go test ./...", DependsOn: []string{"legacy-item"}},
		{ID: "next", Verify: "go test ./...", DependsOn: []string{"legacy-item"}},
	}}}}
	historical, err := advancePlan(document, "item", ".", false)
	if err != nil || len(historical.Items) != 1 || len(historical.Items[0].Steps) != 1 ||
		historical.Items[0].Steps[0].ID != "next" || historical.Items[0].Steps[0].DependsOn[0] != "legacy-item" {
		t.Fatalf("historical literal step and dependency schema changed: %+v, %v", historical, err)
	}
	if _, err := Advance(document, "item", "."); err == nil || !strings.Contains(err.Error(), "has steps") {
		t.Fatalf("live whole-item shortcut accepted an item with steps: %v", err)
	}
	if _, err := Advance(document, "item", "next"); err == nil {
		t.Fatal("live advancement accepted historical dependency syntax")
	}
	if len(document.Items[0].Steps) != 2 || document.Items[0].Steps[0].ID != "." || document.Items[0].Steps[1].ID != "next" {
		t.Fatal("advancement mutated the caller's plan")
	}
}

// TestEnforceCurrentFirstOpenStep pins the shared dispatch rule: the first step, in the plan's order, whose
// prerequisites have left the plan.
func TestEnforceCurrentFirstOpenStep(t *testing.T) {
	t.Parallel()
	p := Plan{Items: []Item{
		{ID: "a", Steps: []Step{{ID: "s1", DependsOn: []string{"b/s2"}}}},
		{ID: "b", Steps: []Step{
			{ID: "s2", DependsOn: []string{"c/s1"}},
			{ID: "s3"},
		}},
		{ID: "c", Steps: []Step{{ID: "s1"}}},
	}}
	it, st, ok := Current(p)
	if !ok || it.ID != "b" || st.ID != "s3" {
		t.Fatalf("Current = %s/%s ok=%v, want b/s3 ok=true", it.ID, st.ID, ok)
	}

	if _, _, ok := Current(Plan{}); ok {
		t.Fatal("Current on an empty plan must return ok=false")
	}

	// An open item with no open step is itself the action (sentinel step ".").
	openNoStep := Plan{Items: []Item{{ID: "x"}}}
	if it, st, ok := Current(openNoStep); !ok || it.ID != "x" || st.ID != "." {
		t.Fatalf("Current(open item) = %s/%s ok=%v, want x/. ok=true", it.ID, st.ID, ok)
	}
}

// TestLivePlanRejectsRetainedCompletionState holds the plan to keeping no completion state: an item and a step are
// open while the plan holds them and leave it when they land (Advance), so neither has a status a plan could keep done
// (single-source/step-fields), and a plan file that names one is refused on read.
func TestLivePlanRejectsRetainedCompletionState(t *testing.T) {
	t.Parallel()
	for _, kept := range []reflect.Type{reflect.TypeFor[Item](), reflect.TypeFor[Step]()} {
		for _, field := range []string{"Status", "Owner"} {
			if _, held := kept.FieldByName(field); held {
				t.Errorf("a plan %s holds a %s", kept.Name(), field)
			}
		}
	}
	if _, err := Parse([]byte(`{"campaign":"x","items":[{"id":"row","steps":[{"id":"do","status":"done","verify":"go test ./..."}]}]}`)); err == nil || !strings.Contains(err.Error(), "status") {
		t.Fatalf("a plan naming a step's status read as %v; want it refused", err)
	}
}

func TestOpenStepRequiresVerifier(t *testing.T) {
	t.Parallel()
	document := Plan{Items: []Item{{
		ID: "item", Steps: []Step{{ID: "work"}},
	}}}
	if err := Validate(document); err == nil {
		t.Fatal("a step without a verifier passed validation")
	}
}

// TestAdvanceRemovesCompletedRows pins the plan contract: the plan
// holds only future, blocked, and in-progress work. Advancing a step
// REMOVES it, the item leaves with its last step, and completion
// history lives in Git through the gate's structured trailers.
func TestAdvanceRemovesCompletedRows(t *testing.T) {
	t.Parallel()
	document := Plan{Items: []Item{{
		ID: "item", Steps: []Step{
			{ID: "first", Verify: "go test ./..."},
			{ID: "second", Verify: "go test ./..."},
		},
	}}}
	advanced := testcheck.Must(Advance(document, "item", "first"))(t)
	if len(advanced.Items[0].Steps) != 1 || advanced.Items[0].Steps[0].ID != "second" {
		t.Fatalf("first advance retained the completed step: %+v", advanced.Items[0])
	}
	if _, step, ok := Current(advanced); !ok || step.ID != "second" {
		t.Fatalf("current after first advance = %s, open=%v", step.ID, ok)
	}
	advanced = testcheck.Must(Advance(advanced, "item", "second"))(t)
	if len(advanced.Items) != 0 {
		t.Fatalf("final advance retained the completed item: %+v", advanced.Items)
	}
	if _, _, ok := Current(advanced); ok {
		t.Fatal("completed plan remained dispatchable")
	}
}
