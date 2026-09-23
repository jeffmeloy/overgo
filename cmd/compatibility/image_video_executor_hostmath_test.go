package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testutil"
)

// mediaExecutorHostmathLayer is the reviewed output-neutral delta of the two
// runtime changes that moved the media source identity past the rotary base:
// the executor device-byte accounting and the host linear-column reorder. Each
// is bound by its before and after source identity plus a receipt that
// re-observes the affected kernels at the current surface; its base is the
// rotary chain's last whole commit.
var mediaExecutorHostmathLayer = mediaLayer{
	name:     "executor/hostmath",
	document: "docs/image_video_executor_hostmath_reconciliation.json",
	sha256:   "d4b8bb18c031b65509a5ac6eae3c421bacd58dfaa32584129c1213165cad9bd6",
	base:     "7abbfa4e29c343f897e56f8eb167c19e0ccae1d4",
}

// checkMediaExecutorHostmathSource peels the executor/hostmath layer; see mediaLayer.
func checkMediaExecutorHostmathSource(root, revision string, paths []string) (string, bool, error) {
	return checkMediaLayerSource(root, revision, paths, mediaExecutorHostmathLayer)
}

// TestMediaExecutorHostmathReconciliation proves the layer peels to its base
// once the newer layers are peeled, and that its document is bound field by field.
func TestMediaExecutorHostmathReconciliation(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	var merged mediaMergedEvidence
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	revision := peelNewerMediaLayers(t, root, merged.RuntimePaths, mediaQ10prefillLayer, mediaWorkflowruntimeLayer)
	requireMediaLayerPeels(t, root, revision, merged.RuntimePaths, mediaExecutorHostmathLayer)
}
