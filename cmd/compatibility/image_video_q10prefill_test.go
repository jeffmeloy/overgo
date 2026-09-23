package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testutil"
)

// mediaQ10prefillLayer is the reviewed output-neutral delta of the Q1_0
// staged-prefill landing above the workflowruntime peel: one added
// dequantise-rows kernel, its manifest and binding entries, and the staged
// table admitting Q1_0. The kernel runs only for Q1_0 weights and no media
// model carries any, so every image and video generation path is byte-for-byte
// the code it was; the bound receipt is the executor suite re-run on the
// device at the current surface, including the F16, BF16 and quantised GEMM
// parity tests the media paths share. Its base is the commit before that
// landing, where the chain below was whole.
var mediaQ10prefillLayer = mediaLayer{
	name:     "q10prefill",
	document: "docs/image_video_q10prefill_reconciliation.json",
	sha256:   "531dbcfdd3d4a35ba11d7cc5068d55537fbb8cd23abc9edacd3bc12c6b58fecb",
	base:     "36c838022b4993050e9386a7b6697d31280a27eb",
}

// checkMediaQ10prefillSource peels the Q1_0 staged-prefill layer; see mediaLayer.
func checkMediaQ10prefillSource(root, revision string, paths []string) (string, bool, error) {
	return checkMediaLayerSource(root, revision, paths, mediaQ10prefillLayer)
}

// TestMediaQ10prefillReconciliation proves the newest layer peels to its base
// at the current surface and that its document is bound field by field.
func TestMediaQ10prefillReconciliation(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	var merged mediaMergedEvidence
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	requireMediaLayerPeels(t, root, "", merged.RuntimePaths, mediaQ10prefillLayer)
}
