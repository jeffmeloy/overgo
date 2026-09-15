package plan

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/worklease"
)

func TestAdvancePreservesHistoricalSchemaAndInput(t *testing.T) {
	document := Plan{Items: []Item{{ID: "item", Status: StatusOpen, Steps: []Step{
		{ID: ".", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"legacy-item"}},
		{ID: "next", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"legacy-item"}},
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
	it, st, ok := Current(p, worklease.UnassignedRole, testCompletionAuthority(t, p))
	if !ok || it.ID != "b" || st.ID != "s2" {
		t.Fatalf("Current = %s/%s ok=%v, want b/s2 ok=true", it.ID, st.ID, ok)
	}

	if _, _, ok := Current(Plan{}, worklease.UnassignedRole, CompletionAuthority{}); ok {
		t.Fatal("Current on an empty plan must return ok=false")
	}

	// An open item with no open step is itself the action (sentinel step ".").
	openNoStep := Plan{Items: []Item{{ID: "x", Status: "open"}}}
	if it, st, ok := Current(openNoStep, worklease.UnassignedRole, testCompletionAuthority(t, openNoStep)); !ok || it.ID != "x" || st.ID != "." {
		t.Fatalf("Current(open item, no steps) = %s/%s ok=%v, want x/. ok=true", it.ID, st.ID, ok)
	}
}

func TestCurrentRefusesUnresolvedOrCrossPlanAuthority(t *testing.T) {
	document := Plan{Items: []Item{{
		ID: "one", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	if _, _, open := Current(document, worklease.UnassignedRole, CompletionAuthority{}); open {
		t.Fatal("zero completion authority dispatched work")
	}
	other := Plan{Items: []Item{{
		ID: "other", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}}}
	if _, _, open := Current(document, worklease.UnassignedRole, testCompletionAuthority(t, other)); open {
		t.Fatal("completion authority was replayed across plans")
	}
}

func TestRoleOwnedDispatch(t *testing.T) {
	document := Plan{Items: []Item{
		{ID: "shared", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
		{ID: "developer", Owner: "developer", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
		{ID: "sqa", Owner: "sqa", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
	}}
	authority := testCompletionAuthority(t, document)
	item, _, ok := Current(document, "sqa", authority)
	if !ok || item.ID != "sqa" {
		t.Fatalf("sqa dispatch = %s, ok=%v", item.ID, ok)
	}
	item, _, ok = Current(document, "developer", authority)
	if !ok || item.ID != "developer" {
		t.Fatalf("developer dispatch = %s, ok=%v", item.ID, ok)
	}
}

func TestUnownedDispatchFallback(t *testing.T) {
	document := Plan{Items: []Item{
		{ID: "shared", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
		{ID: "developer", Owner: "developer", Status: "open", Steps: []Step{{ID: "do", Status: "open"}}},
	}}
	authority := testCompletionAuthority(t, document)
	for _, role := range []string{worklease.UnassignedRole, "sqa"} {
		item, _, ok := Current(document, role, authority)
		if !ok || item.ID != "shared" {
			t.Fatalf("fallback for %s = %s, ok=%v", role, item.ID, ok)
		}
	}
	foreignOnly := Plan{Items: document.Items[1:]}
	if _, _, ok := Current(foreignOnly, "sqa", testCompletionAuthority(t, foreignOnly)); ok {
		t.Fatal("foreign owned work was dispatched")
	}
}

func TestLivePlanRejectsRetainedCompletionState(t *testing.T) {
	valid := Plan{Items: []Item{
		{ID: "open", Status: "open", Steps: []Step{{ID: "work", Status: "open", Verify: "go test ./..."}}},
		{ID: "blocked", Status: "blocked-external-prereq", Steps: []Step{{ID: "wait", Status: "blocked-external-prereq"}}},
	}}
	if err := Validate(valid); err != nil {
		t.Fatal(err)
	}
	for name, invalid := range map[string]Plan{
		"done item": {Items: []Item{{
			ID: "done", Status: StatusDone,
			Steps: []Step{{ID: "old", Status: StatusDone, Verify: "go test ./..."}},
		}}},
		"done step": {Items: []Item{{
			ID: "mixed", Status: StatusOpen,
			Steps: []Step{{ID: "old", Status: StatusDone, Verify: "go test ./..."}},
		}}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := Validate(invalid); err == nil || !strings.Contains(err.Error(), "retains completed state") {
				t.Fatalf("retained done row validation error = %v", err)
			}
		})
	}
}

func TestParseRejectsRawDoneFlip(t *testing.T) {
	for name, raw := range map[string]string{
		"item": `{"campaign":"x","doctrine":"x","items":[{"id":"row","status":"done","steps":[]}]}`,
		"step": `{"campaign":"x","doctrine":"x","items":[{"id":"row","status":"open","steps":[{"id":"do","status":"done","verify":"go test ./..."}]}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse([]byte(raw)); err == nil || !strings.Contains(err.Error(), "retains completed state") {
				t.Fatalf("raw done flip error = %v", err)
			}
			if _, err := ParseHistorical([]byte(raw)); err != nil {
				t.Fatalf("historical done row was not readable: %v", err)
			}
		})
	}
	if _, err := ParseHistorical([]byte(`{"items":[],"unknown":true}`)); err == nil {
		t.Fatal("historical parser accepted an unknown field")
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

// TestAdvanceRemovesCompletedRows pins the plan contract: the plan
// holds only future, blocked, and in-progress work. Advancing a step
// REMOVES it, the item leaves with its last step, and completion
// history lives in Git through the gate's structured trailers.
func TestAdvanceRemovesCompletedRows(t *testing.T) {
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
	if len(advanced.Items[0].Steps) != 1 || advanced.Items[0].Steps[0].ID != "second" {
		t.Fatalf("first advance retained the completed step: %+v", advanced.Items[0])
	}
	if _, step, ok := Current(advanced, worklease.UnassignedRole, testCompletionAuthority(t, advanced)); !ok || step.ID != "second" {
		t.Fatalf("current after first advance = %s, open=%v", step.ID, ok)
	}
	advanced, err = Advance(advanced, "item", "second")
	if err != nil {
		t.Fatal(err)
	}
	if len(advanced.Items) != 0 {
		t.Fatalf("final advance retained the completed item: %+v", advanced.Items)
	}
	if _, _, ok := Current(advanced, worklease.UnassignedRole, testCompletionAuthority(t, advanced)); ok {
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
