package plan

import (
	"strings"
	"testing"
)

func dependencyFixture() Plan {
	return Plan{
		Campaign: "dependency-fixture",
		Doctrine: "fixture doctrine",
		Items: []Item{
			{ID: "root", Status: StatusOpen, Owner: "gui", Steps: []Step{{
				ID: "do", Title: "root work", Status: StatusOpen, Verify: "exit 0",
			}}},
			{ID: "dependent", Status: StatusOpen, Owner: "gui", Steps: []Step{{
				ID: "do", Title: "dependent work", Status: StatusOpen, Verify: "exit 0",
				DependsOn: []string{"root/do"},
			}}},
		},
	}
}

// TestDependencyDispatchSkipsBlocked pins enforcement: a step whose
// dependencies are not done never dispatches, whatever the file order
// says, and dispatches as soon as they are.
func TestDependencyDispatchSkipsBlocked(t *testing.T) {
	document := dependencyFixture()
	document.Items[0], document.Items[1] = document.Items[1], document.Items[0]
	item, _, ok := Current(document, "gui", testCompletionAuthority(t, document))
	if !ok || item.ID != "root" {
		t.Fatalf("dispatch = (%s, %t), want the unblocked root despite file order", item.ID, ok)
	}
	// Completion removes the root row entirely, but absence alone is not
	// evidence. The derived completion authority must name the exact row.
	document.Items = document.Items[:1]
	if _, _, ok := Current(document, "gui", CompletionAuthority{}); ok {
		t.Fatal("dependent dispatched from absence without completion authority")
	}
	authority := testCompletionAuthority(t, document, "root/do")
	item, _, ok = Current(document, "gui", authority)
	if !ok || item.ID != "dependent" {
		t.Fatalf("dispatch = (%s, %t), want the dependent once its root completed and left", item.ID, ok)
	}
	// A reference to a PRESENT but unfinished sibling step still blocks.
	document.Items[0].Steps = append(document.Items[0].Steps,
		Step{ID: "later", Status: StatusOpen, Verify: "go test ./..."})
	document.Items[0].Steps[0].DependsOn = []string{"dependent/later"}
	authority = testCompletionAuthority(t, document, "root/do")
	if item, step, ok := Current(document, "gui", authority); !ok || step.ID != "later" {
		t.Fatalf("dispatch = (%s/%s, %t), want the unblocked sibling", item.ID, step.ID, ok)
	}
}

// TestPrunedDependencyRequiresExplicitAuthority pins the fail-closed scheduler
// boundary: a syntactically valid missing row never satisfies depends_on until
// the opaque, derived authority carries that exact completion.
func TestPrunedDependencyRequiresExplicitAuthority(t *testing.T) {
	document := dependencyFixture()
	document.Items = document.Items[1:]
	if _, _, open := Current(document, "gui", CompletionAuthority{}); open {
		t.Fatal("unknown pruned dependency dispatched")
	}
	wrong := testCompletionAuthority(t, document, "other/do")
	if _, _, open := Current(document, "gui", wrong); open {
		t.Fatal("foreign completion evidence satisfied dependency")
	}
	exact := testCompletionAuthority(t, document, "root/do")
	if item, step, open := Current(document, "gui", exact); !open || item.ID != "dependent" || step.ID != "do" {
		t.Fatalf("exact completion did not dispatch dependent: %s/%s open=%v", item.ID, step.ID, open)
	}
}

// TestDependencyValidationRefusesCycles pins the load-time contract:
// the dependency graph must be acyclic among PRESENT rows; a reference
// to an absent row remains representable because completion removes rows,
// while dispatch separately requires its completion authority.
func TestDependencyValidationRefusesCycles(t *testing.T) {
	valid := dependencyFixture()
	if err := Validate(valid); err != nil {
		t.Fatalf("valid dependency graph refused: %v", err)
	}
	completed := dependencyFixture()
	completed.Items[1].Steps[0].DependsOn = []string{"ghost/do"}
	if err := Validate(completed); err != nil {
		t.Fatalf("reference to a completed-and-removed row refused: %v", err)
	}
	cyclic := dependencyFixture()
	cyclic.Items[0].Steps[0].DependsOn = []string{"dependent/do"}
	if err := Validate(cyclic); err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("dependency cycle accepted: %v", err)
	}
}

func TestDependencyValidationRequiresExactSteps(t *testing.T) {
	for _, reference := range []string{"root", "/do", "root/", "root/do/extra", " root/do", "root/ do", "root\\do"} {
		document := dependencyFixture()
		document.Items[1].Steps[0].DependsOn = []string{reference}
		if err := Validate(document); err == nil || !strings.Contains(err.Error(), "exact item/step") {
			t.Errorf("depends_on %q error = %v, want exact-step refusal", reference, err)
		}
	}
	completed := dependencyFixture()
	completed.Items[1].Steps[0].DependsOn = []string{"completed/do", "root/completed-step"}
	if err := Validate(completed); err != nil {
		t.Fatalf("exact pruned step refused: %v", err)
	}
	for _, mutate := range []func(*Plan){
		func(document *Plan) { document.Items[0].ID = "root/item" },
		func(document *Plan) { document.Items[0].Steps[0].ID = "bad step" },
	} {
		document := dependencyFixture()
		mutate(&document)
		if err := Validate(document); err == nil || !strings.Contains(err.Error(), "invalid") {
			t.Errorf("malformed plan identity accepted: %v", err)
		}
	}
}

// TestDependencySameItemEdges pins the same-item contract: a step may depend
// on a sibling step, but never on itself, and mutually dependent siblings are
// a cycle.
func TestDependencySameItemEdges(t *testing.T) {
	sibling := dependencyFixture()
	sibling.Items[0].Steps = append(sibling.Items[0].Steps, Step{
		ID: "verify", Title: "verify root", Status: StatusOpen, Verify: "exit 0",
		DependsOn: []string{"root/do"},
	})
	if err := Validate(sibling); err != nil {
		t.Fatalf("sibling step dependency refused: %v", err)
	}
	self := dependencyFixture()
	self.Items[0].Steps[0].DependsOn = []string{"root/do"}
	if err := Validate(self); err == nil || !strings.Contains(err.Error(), "itself") {
		t.Fatalf("self step dependency accepted: %v", err)
	}
	mutual := dependencyFixture()
	mutual.Items[1].Steps[0].DependsOn = nil
	mutual.Items[0].Steps = []Step{
		{ID: "a", Title: "a", Status: StatusOpen, Verify: "exit 0", DependsOn: []string{"root/b"}},
		{ID: "b", Title: "b", Status: StatusOpen, Verify: "exit 0", DependsOn: []string{"root/a"}},
	}
	if err := Validate(mutual); err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("mutually dependent siblings accepted: %v", err)
	}
}
