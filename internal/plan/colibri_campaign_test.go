package plan

import (
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// The Colibri lane has its own queue. Master obligations are retained only
// across bootstrap, until the existing gated lane projection can omit them
// without claiming completion; first-parent-target merges preserve this queue.
func assertColibriCampaign(t *testing.T, document Plan) {
	t.Helper()
	testcheck.NoError(t, Validate(document))
	// A lane's plan holds its doctrine as one string or as paragraphs, read as an earlier plan's; either form is searched.
	doctrine := string(document.DoctrineBefore) + string(document.ParagraphsBefore)
	for _, decision := range []string{
		"first-parent-target", "end-to-end", "held-out", "common components",
		"caller-declared budgets", "no completion credit", "cmd/gate",
	} {
		if !strings.Contains(doctrine, decision) {
			t.Errorf("Colibri doctrine omits %q", decision)
		}
	}
	for _, item := range document.Items {
		if item.Class == ClassMerge {
			if !preparedColibriMergeBoundary(document.Lane, item) {
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

// First-parent-target lane merges retain the build verifier emitted by
// cmd/plan.mergeVerify. A merge may add a differential named test before it.
// The full compatibility verifier remains mandatory for non-lane merges. The lane is the plan's: every item is its lane's.
func preparedColibriMergeBoundary(lane string, item Item) bool {
	if (lane != "colibri" && lane != "colibri2") || !preparedMergeShape(item) {
		return false
	}
	verify := item.Steps[0].Verify
	return verify == "go build ./..." ||
		(strings.HasPrefix(verify, "go test ./") && strings.HasSuffix(verify, " && go build ./..."))
}

func TestColibriPreparedMergeBoundary(t *testing.T) {
	t.Parallel()
	item := Item{ID: "merge-939317c05122",
		Steps: []Step{{ID: "do", Verify: "go build ./..."}}}
	if !preparedColibriMergeBoundary("colibri", item) || preparedMergeBoundary(item) {
		t.Fatal("lane build boundary must be accepted only as a lane merge")
	}
	if !preparedColibriMergeBoundary("colibri2", item) || preparedMergeBoundary(item) {
		t.Fatal("Colibri2 build boundary must be accepted only as a lane merge")
	}
	if preparedColibriMergeBoundary("master-lead", item) {
		t.Fatal("a merge of another lane's plan was accepted as a Colibri merge")
	}
	item.Steps[0].Verify = "go test ./internal/runrecord -run ^TestParseVerify$ -count=1 && go build ./..."
	if !preparedColibriMergeBoundary("colibri", item) {
		t.Fatal("lane merge must admit a differential named test followed by the build")
	}
	item.Steps[0].Verify = "go build ./..."
	for _, mutate := range []func(*Item){
		func(item *Item) { item.ID = "merge-not-a-revision" },
		func(item *Item) { item.Steps = nil },
		func(item *Item) {
			item.Steps = []Step{{ID: "do", Verify: "go build ./... || true"}}
		},
	} {
		invalid := item
		mutate(&invalid)
		if preparedColibriMergeBoundary("colibri", invalid) {
			t.Fatalf("accepted invalid lane merge %+v", invalid)
		}
	}
}

func TestColibriCampaign(t *testing.T) {
	t.Parallel()
	document := loadCampaignPlan(t)
	if document.Lane == "colibri" || document.Lane == "colibri2" {
		assertColibriCampaign(t, document)
	}
}
