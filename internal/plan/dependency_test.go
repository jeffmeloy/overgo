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
	item, _, ok := Current(document, "gui")
	if !ok || item.ID != "root" {
		t.Fatalf("dispatch = (%s, %t), want the unblocked root despite file order", item.ID, ok)
	}
	document.Items[1].Status = StatusDone
	document.Items[1].Steps[0].Status = StatusDone
	item, _, ok = Current(document, "gui")
	if !ok || item.ID != "dependent" {
		t.Fatalf("dispatch = (%s, %t), want the dependent once its root is done", item.ID, ok)
	}
	document.Items[0].Steps[0].DependsOn = []string{"missing-item/do"}
	if _, _, ok := Current(document, "gui"); ok {
		t.Fatal("a step with an unsatisfiable dependency dispatched")
	}
}

// TestDependencyValidationRefusesDanglingAndCycles pins the load-time
// contract: references must resolve and the dependency graph must be
// acyclic.
func TestDependencyValidationRefusesDanglingAndCycles(t *testing.T) {
	valid := dependencyFixture()
	if err := Validate(valid); err != nil {
		t.Fatalf("valid dependency graph refused: %v", err)
	}
	dangling := dependencyFixture()
	dangling.Items[1].Steps[0].DependsOn = []string{"ghost/do"}
	if err := Validate(dangling); err == nil || !strings.Contains(err.Error(), "unknown item") {
		t.Fatalf("dangling item reference accepted: %v", err)
	}
	wrongStep := dependencyFixture()
	wrongStep.Items[1].Steps[0].DependsOn = []string{"root/undo"}
	if err := Validate(wrongStep); err == nil || !strings.Contains(err.Error(), "unknown step") {
		t.Fatalf("dangling step reference accepted: %v", err)
	}
	cyclic := dependencyFixture()
	cyclic.Items[0].Steps[0].DependsOn = []string{"dependent/do"}
	if err := Validate(cyclic); err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("dependency cycle accepted: %v", err)
	}
}
