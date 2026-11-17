package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheCampaignLineIsSetByTheTool holds the plan's campaign line to the tool that sets it (campaign-line/set): -campaign
// -title sets the line its rows serve, trimmed, whatever rows the plan holds, an empty line refused; and the plan names
// the owner's campaign of 2026-11-17, one supplied workflow, the MOSS Ultra workflow, each section held to ComfyUI's
// recording.
func TestTheCampaignLineIsSetByTheTool(t *testing.T) {
	t.Parallel()
	held := plan.Plan{Campaign: "old", Items: []plan.Item{{ID: "row", Title: "row", Steps: []plan.Step{{ID: "do", Title: "row"}}}}}
	set := testcheck.Must(setCampaign(held, "  one workflow, each section held to its recording  "))(t)
	if set.Campaign != "one workflow, each section held to its recording" || len(set.Items) != 1 {
		t.Fatalf("the campaign line set over a plan with rows reads %q, its rows %d", set.Campaign, len(set.Items))
	}
	if _, err := setCampaign(held, " "); err == nil {
		t.Fatal("an empty campaign line was set")
	}
	document := testcheck.Must(plan.Load(filepath.Join(testutil.RepoRoot(t), filepath.FromSlash(plan.Path))))(t)
	if !strings.Contains(document.Campaign, "MOSS Ultra workflow") || !strings.Contains(document.Campaign, "recording") {
		t.Fatalf("the plan's campaign line is %q, not the owner's MOSS Ultra workflow held to ComfyUI's recordings", document.Campaign)
	}
}
