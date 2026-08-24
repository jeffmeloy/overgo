package plan

import (
	"testing"

	"overgo/internal/artifact"
)

// TestEnforceCurrentFirstOpenStep pins the shared dispatch rule.
func TestEnforceCurrentFirstOpenStep(t *testing.T) {
	p := Plan{Items: []Item{
		{ID: "a", Status: StatusDone, Steps: []Step{{ID: "s1", Status: StatusDone}}},
		{ID: "b", Status: "open", Steps: []Step{
			{ID: "s1", Status: StatusDone},
			{ID: "s2", Status: "open"},
		}},
		{ID: "c", Status: "open", Steps: []Step{{ID: "s1", Status: "open"}}},
	}}
	it, st, ok := Current(p, UnassignedRole)
	if !ok || it.ID != "b" || st.ID != "s2" {
		t.Fatalf("Current = %s/%s ok=%v, want b/s2 ok=true", it.ID, st.ID, ok)
	}

	if _, _, ok := Current(Plan{}, UnassignedRole); ok {
		t.Fatal("Current on an empty plan must return ok=false")
	}

	// An open item with no open step is itself the action (sentinel step ".").
	openNoStep := Plan{Items: []Item{{ID: "x", Status: "open"}}}
	if it, st, ok := Current(openNoStep, UnassignedRole); !ok || it.ID != "x" || st.ID != "." {
		t.Fatalf("Current(open item, no steps) = %s/%s ok=%v, want x/. ok=true", it.ID, st.ID, ok)
	}
}

func TestRoleOwnedDispatch(t *testing.T) {
	document := Plan{Items: []Item{
		{ID: "shared", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
		{ID: "developer", Owner: "developer", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
		{ID: "sqa", Owner: "sqa", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
	}}
	item, _, ok := Current(document, "sqa")
	if !ok || item.ID != "sqa" {
		t.Fatalf("sqa dispatch = %s, ok=%v", item.ID, ok)
	}
	item, _, ok = Current(document, "developer")
	if !ok || item.ID != "developer" {
		t.Fatalf("developer dispatch = %s, ok=%v", item.ID, ok)
	}
}

func TestUnownedDispatchFallback(t *testing.T) {
	document := Plan{Items: []Item{
		{ID: "shared", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
		{ID: "developer", Owner: "developer", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
	}}
	for _, role := range []string{UnassignedRole, "sqa"} {
		item, _, ok := Current(document, role)
		if !ok || item.ID != "shared" {
			t.Fatalf("fallback for %s = %s, ok=%v", role, item.ID, ok)
		}
	}
	if _, _, ok := Current(Plan{Items: document.Items[1:]}, "sqa"); ok {
		t.Fatal("foreign owned work was dispatched")
	}
}

func TestPlanRetainsCompletionState(t *testing.T) {
	valid := Plan{Items: []Item{
		{ID: "open", Status: "open", Steps: []Step{{ID: "work", Status: "open", Verify: "go test ./..."}}},
		{ID: "blocked", Status: "blocked-external-prereq", Steps: []Step{{ID: "wait", Status: "blocked-external-prereq"}}},
		{ID: "done", Status: StatusDone, Steps: []Step{{ID: "old", Status: StatusDone, Verify: "go test ./..."}}},
	}}
	if err := Validate(valid); err != nil {
		t.Fatal(err)
	}
	invalid := Plan{Items: []Item{
		{ID: "mixed", Status: StatusDone, Steps: []Step{{ID: "next", Status: StatusOpen, Verify: "go test ./..."}}},
	}}
	if err := Validate(invalid); err == nil {
		t.Fatal("done item with open work passed validation")
	}
}

func TestOpenStepRequiresVerifier(t *testing.T) {
	document := Plan{Items: []Item{{
		ID: "item", Status: "open", Steps: []Step{{ID: "work", Status: "open"}},
	}}}
	if err := Validate(document); err == nil {
		t.Fatal("open step without a verifier passed validation")
	}
	document.Items[0].Steps[0].Status = "blocked-external-prereq"
	if err := Validate(document); err != nil {
		t.Fatalf("blocked step should name its blocker without a runnable verifier: %v", err)
	}
}

func TestAdvanceMarksRowsDone(t *testing.T) {
	document := Plan{Items: []Item{{
		ID: "item", Status: StatusOpen, Steps: []Step{
			{ID: "first", Status: StatusOpen, Verify: "go test ./..."},
			{ID: "second", Status: StatusOpen, Verify: "go test ./..."},
		},
	}}}
	advanced, err := Advance(document, "item", "first")
	if err != nil {
		t.Fatal(err)
	}
	if len(advanced.Items[0].Steps) != len(document.Items[0].Steps) || advanced.Items[0].Steps[0].Status != StatusDone {
		t.Fatalf("first advance removed history: %+v", advanced.Items[0])
	}
	if _, step, ok := Current(advanced, UnassignedRole); !ok || step.ID != "second" {
		t.Fatalf("current after first advance = %s, open=%v", step.ID, ok)
	}
	advanced, err = Advance(advanced, "item", "second")
	if err != nil {
		t.Fatal(err)
	}
	if advanced.Items[0].Status != StatusDone || advanced.Items[0].Steps[1].Status != StatusDone {
		t.Fatalf("final advance = %+v", advanced.Items[0])
	}
	if _, _, ok := Current(advanced, UnassignedRole); ok {
		t.Fatal("completed plan remained dispatchable")
	}
}

func TestCampaignCensusAuthority(t *testing.T) {
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("census"))
	if err != nil {
		t.Fatal(err)
	}
	document := Plan{Campaign: "campaign", Doctrine: "Measured baselines live in OvergoDB.", Census: &id}
	if err := ValidateCampaignCensusAuthority(document); err != nil {
		t.Fatal(err)
	}
	document.Census = nil
	if err := ValidateCampaignCensusAuthority(document); err == nil {
		t.Fatal("missing census authority passed")
	}
}

func TestDoctrineRejectsMetricLiterals(t *testing.T) {
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("census"))
	if err != nil {
		t.Fatal(err)
	}
	document := Plan{Campaign: "campaign", Doctrine: "Measured baseline: 18,277 literals and 869 policy copies.", Census: &id}
	if err := ValidateCampaignCensusAuthority(document); err == nil {
		t.Fatal("hand-typed doctrine metrics passed")
	}
}
