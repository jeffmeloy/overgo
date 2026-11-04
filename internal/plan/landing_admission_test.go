package plan

import (
	"testing"

	"overgo/internal/worklease"
)

// TestAWorkerLandsOnlyItsDispatchedStep holds a landing to what dispatch would give its role. Any session is refused
// a step whose prerequisite is open and another lane's row, and may land any ready step of its own. A worker is
// refused all but the step dispatch names, and a batch's later steps count once its earlier ones have left the plan.
func TestAWorkerLandsOnlyItsDispatchedStep(t *testing.T) {
	t.Parallel()
	step := func(id string, after ...string) Step {
		return Step{ID: id, Status: StatusOpen, Verify: "go test ./...", DependsOn: after}
	}
	document := Plan{Items: []Item{
		{ID: "first", Status: StatusOpen, Steps: []Step{step("one"), step("two")}},
		{ID: "later", Status: StatusOpen, Steps: []Step{step("do", "first/two")}},
		{ID: "lane", Status: StatusOpen, Owner: "other-lane", Steps: []Step{step("do")}},
	}}
	role := worklease.UnassignedRole
	for _, check := range []struct {
		worker   bool
		refs     []Ref
		admitted bool
	}{
		{false, []Ref{{"first", "two"}}, true},
		{false, []Ref{{"later", "do"}}, false},
		{false, []Ref{{"lane", "do"}}, false},
		{true, []Ref{{"first", "one"}}, true},
		{true, []Ref{{"first", "two"}}, false},
		{true, []Ref{{"first", "one"}, {"first", "two"}, {"later", "do"}}, true},
		{true, []Ref{{"first", "one"}, {"later", "do"}}, false},
	} {
		current, err := document, error(nil)
		for _, ref := range check.refs {
			if err = admits(current, role, check.worker, ref); err != nil {
				break
			}
			if current, err = advancePlan(current, ref.Item, ref.Step, false); err != nil {
				t.Fatal(err)
			}
		}
		if admitted := err == nil; admitted != check.admitted {
			t.Errorf("worker=%v landing %v: admitted=%v (%v), want %v", check.worker, check.refs, admitted, err, check.admitted)
		}
	}
}
