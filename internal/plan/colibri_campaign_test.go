package plan

import (
	"strings"
	"testing"
)

// The Colibri lane has its own queue. Master obligations are retained only
// across bootstrap, until the existing gated lane projection can omit them
// without claiming completion; first-parent-target merges preserve this queue.
func assertColibriCampaign(t *testing.T, document Plan) {
	t.Helper()
	if err := Validate(document); err != nil {
		t.Fatal(err)
	}
	for _, decision := range []string{
		"first-parent-target", "end-to-end", "held-out", "common components",
		"caller-declared budgets", "no completion credit", "cmd/gate",
	} {
		if !strings.Contains(document.Doctrine, decision) {
			t.Errorf("Colibri doctrine omits %q", decision)
		}
	}
	for _, item := range document.Items {
		if item.Owner != document.Lane {
			if document.Scope == ScopeLane || item.Owner == "" {
				t.Errorf("Colibri dispatch includes unowned or foreign item %s", item.ID)
			}
			continue
		}
		if strings.HasPrefix(item.ID, MergeItemPrefix) {
			if !preparedColibriMergeBoundary(item) {
				t.Errorf("invalid Colibri merge boundary %s", item.ID)
			}
			continue
		}
		for _, step := range item.Steps {
			if strings.TrimSpace(step.Rationale) == "" || len(step.Capabilities) == 0 {
				t.Errorf("Colibri step %s/%s lacks rationale or capability contract", item.ID, step.ID)
			}
		}
	}
}

// first-parent-target lane merges use the build verifier emitted by
// cmd/plan.mergeVerify. The gate separately proves the parent receipts;
// the full compatibility verifier remains mandatory for non-lane merges.
func preparedColibriMergeBoundary(item Item) bool {
	return item.Owner == "colibri" && preparedMergeShape(item) && item.Steps[0].Verify == "go build ./..."
}

func TestColibriPreparedMergeBoundary(t *testing.T) {
	item := Item{ID: "merge-939317c05122", Owner: "colibri", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go build ./..."}}}
	if !preparedColibriMergeBoundary(item) || preparedMergeBoundary(item) {
		t.Fatal("lane build boundary must be accepted only as a lane merge")
	}
	for _, mutate := range []func(*Item){
		func(item *Item) { item.Owner = "master-lead" },
		func(item *Item) { item.ID = "merge-not-a-revision" },
		func(item *Item) { item.Status = StatusDone },
		func(item *Item) { item.Steps = nil },
		func(item *Item) {
			item.Steps = []Step{{ID: "do", Status: StatusOpen, Verify: "go build ./... || true"}}
		},
	} {
		invalid := item
		mutate(&invalid)
		if preparedColibriMergeBoundary(invalid) {
			t.Fatalf("accepted invalid lane merge %+v", invalid)
		}
	}
}

func TestColibriCampaign(t *testing.T) {
	document := loadCampaignPlan(t)
	if document.Lane == "colibri" {
		assertColibriCampaign(t, document)
	}
}

func TestColibriLaneRejectsForeignWork(t *testing.T) {
	document := Plan{Lane: "colibri", Scope: ScopeLane, Items: []Item{{
		ID: "foreign", Owner: "master-lead", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./internal/plan"}},
	}}}
	if err := Validate(document); err == nil {
		t.Fatal("Colibri lane admitted another lane's work")
	}
}
