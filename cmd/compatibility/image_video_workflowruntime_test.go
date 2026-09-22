package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testutil"
)

// mediaWorkflowruntimeLayer is the reviewed output-neutral delta of the two
// harness landings under internal/workflowruntime after the executor/hostmath
// base: an idempotent commit routed through its one owner, and a dead wrapper
// removed. Neither touches a generation path; the media runtime paths include
// the package because the media lifecycle workflow runs through it, and the
// receipt re-observes that workflow's own tests at the current surface. Its
// base is the commit at which the chain was last whole before those landings.
var mediaWorkflowruntimeLayer = mediaLayer{
	name:     "workflowruntime",
	document: "docs/image_video_workflowruntime_reconciliation.json",
	sha256:   "6ea7f37c4f6096b37a3058dd9a5a171e66400e650c2b52ab77667919e30311d4",
	base:     "043805ef7f349dd94c1d2e6a7224fe0ccef34ef8",
}

// checkMediaWorkflowruntimeSource peels the workflowruntime layer; see mediaLayer.
func checkMediaWorkflowruntimeSource(root, revision string, paths []string) (string, bool, error) {
	return checkMediaLayerSource(root, revision, paths, mediaWorkflowruntimeLayer)
}

// TestMediaWorkflowruntimeReconciliation proves the layer peels to its base
// once the newer layer is peeled, and that its document is bound field by field.
func TestMediaWorkflowruntimeReconciliation(t *testing.T) {
	root := testutil.RepoRoot(t)
	var merged mediaMergedEvidence
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	paths := append([]string{"internal/workflowruntime"}, merged.RuntimePaths...)
	revision := peelNewerMediaLayers(t, root, paths, mediaQ10prefillLayer)
	requireMediaLayerPeels(t, root, revision, paths, mediaWorkflowruntimeLayer)
}
