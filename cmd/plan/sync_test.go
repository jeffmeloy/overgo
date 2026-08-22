package main

import (
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
	local := plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: []plan.Item{item("base-upstream"), item("local-new")}}
	upstream := plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: []plan.Item{item("base-local"), item("upstream-new")}}
	merged, err := plan.MergeOpenProjections(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(merged.Items))
	for index, mergedItem := range merged.Items {
		ids[index] = mergedItem.ID
	}
	if !slices.Equal(ids, []string{"local-new", "upstream-new"}) {
		t.Fatalf("merged open projection = %v", ids)
	}
	merged, err = insertItem(merged, "merge-0123456789ab", "Merge topic at 0123456789ab", "", "go run ./cmd/compatibility -check")
	if err != nil || merged.Items[0].ID != "merge-0123456789ab" || merged.Items[0].Steps[0].Verify != "go run ./cmd/compatibility -check" {
		t.Fatalf("prepared merge plan = %+v, %v", merged.Items[0], err)
	}
	local = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	upstream = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	local.Items[0].Title = "local edit"
	upstream.Items[0].Title = "upstream edit"
	if _, err := plan.MergeOpenProjections(base, local, upstream); err == nil {
		t.Fatal("concurrent edits to one live plan row were guessed")
	}
}
