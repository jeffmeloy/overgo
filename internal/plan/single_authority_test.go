package plan

import (
	"os"
	"path/filepath"
	"testing"
)

// TestSinglePlanAuthority pins the owner directive (2026-08-16): one skill.md
// doctrine and one plan.json dispatch authority. The campaign stalled once
// because open rows lived in a second plan document the dispatcher never read;
// this test refuses any reintroduction of a parallel plan surface.
func TestSinglePlanAuthority(t *testing.T) {
	t.Parallel()
	// The tree is found from the directory the test runs in, as loadCampaignPlan finds it.
	root := filepath.Join("..", "..")
	for _, banned := range []string{
		"docs/rsi_plan.json",
		"cmd/roadmap",
	} {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(banned))); !os.IsNotExist(err) {
			t.Errorf("parallel plan surface exists: %s (stat err %v)", banned, err)
		}
	}
	document, err := Load(filepath.Join(root, filepath.FromSlash(Path)))
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(document); err != nil {
		t.Fatal(err)
	}
}
