package plan

import "testing"

func TestMergeCompletedCampaignProjections(t *testing.T) {
	base := Plan{Campaign: "base", Doctrine: "base doctrine"}
	local := Plan{Campaign: "completed local", Doctrine: "local doctrine"}
	upstream := Plan{Campaign: "completed upstream", Doctrine: "upstream doctrine"}
	merged, err := MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	if merged.Campaign != upstream.Campaign || merged.Doctrine != upstream.Doctrine || len(merged.Items) != 0 {
		t.Fatalf("completed projections = %+v", merged)
	}
	local.Items = []Item{{
		ID: "merge", Title: "merge", Status: StatusOpen,
		Steps: []Step{{ID: "do", Title: "merge", Status: StatusOpen, Verify: "go test ./internal/plan"}},
	}}
	merged, err = MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	if merged.Campaign != local.Campaign || len(merged.Items) != 1 || merged.Items[0].ID != "merge" {
		t.Fatalf("active projection = %+v", merged)
	}
}
