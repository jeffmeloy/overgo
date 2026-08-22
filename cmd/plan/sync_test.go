package main

import (
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/plan"
)

func TestPrepareMergeSnapshotsSourceAndRegeneratesDerivedDocs(t *testing.T) {
	item := func(id string) plan.Item {
		return plan.Item{ID: id, Title: id, Status: "open", Steps: []plan.Step{{
			ID: "do", Title: id, Status: "open", Verify: "go test ./...",
		}}}
	}
	base := plan.Plan{Campaign: "campaign", Doctrine: "doctrine", Items: []plan.Item{item("base-local"), item("base-upstream")}}
	local := plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: []plan.Item{item("base-local"), item("base-upstream"), item("local-new")}}
	upstream := plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: []plan.Item{item("base-local"), item("base-upstream"), item("upstream-new")}}
	local.Items[0].Status = plan.StatusDone
	local.Items[0].Steps[0].Status = plan.StatusDone
	upstream.Items[1].Status = plan.StatusDone
	upstream.Items[1].Steps[0].Status = plan.StatusDone
	merged, err := plan.MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(merged.Items))
	for index, mergedItem := range merged.Items {
		ids[index] = mergedItem.ID
	}
	if !slices.Equal(ids, []string{"base-local", "base-upstream", "local-new", "upstream-new"}) ||
		merged.Items[0].Status != plan.StatusDone || merged.Items[1].Status != plan.StatusDone {
		t.Fatalf("merged retained document = %v", merged.Items)
	}
	merged, err = insertItem(merged, "merge-0123456789ab", "Merge topic at 0123456789ab", "", "go run ./cmd/compatibility -check")
	if err != nil || merged.Items[0].ID != "merge-0123456789ab" || merged.Items[0].Steps[0].Verify != "go run ./cmd/compatibility -check" {
		t.Fatalf("prepared merge plan = %+v, %v", merged.Items[0], err)
	}
	local = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	upstream = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	local.Items[0].Title = "local edit"
	upstream.Items[0].Title = "upstream edit"
	if _, err := plan.MergeDocuments(base, local, upstream); err == nil {
		t.Fatal("concurrent edits to one live plan row were guessed")
	}

	local = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	upstream = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	local.Items = local.Items[1:]
	if _, err := plan.MergeDocuments(base, local, upstream); err == nil {
		t.Fatal("legacy row deletion was accepted as completion")
	}

	base.Items[0].Steps[0].Rationale = "why"
	base.Items[0].Steps[0].DependsOn = []string{"prior"}
	base.Items[0].Steps[0].Capabilities = []string{"runtime"}
	base.Items[0].Steps[0].Outcome = json.RawMessage(`{"verdict":"pass"}`)
	local = clonePlan(t, base)
	upstream = clonePlan(t, base)
	local.Items[0].Steps[0].Status = plan.StatusDone
	local.Items[0].Status = plan.StatusDone
	merged, err = plan.MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	step := merged.Items[0].Steps[0]
	if step.Rationale != "why" || !slices.Equal(step.DependsOn, []string{"prior"}) ||
		!slices.Equal(step.Capabilities, []string{"runtime"}) || string(step.Outcome) != `{"verdict":"pass"}` {
		t.Fatalf("merged step lost fields: %+v", step)
	}
}

func clonePlan(t *testing.T, document plan.Plan) plan.Plan {
	t.Helper()
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	cloned, err := plan.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	return cloned
}
