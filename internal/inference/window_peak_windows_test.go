//go:build windows

package inference

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

// TestACellRunnerDecodesPlainlyOutsideWindows holds a qwen35 runner over attention cells, which verifies its drafts in
// Strata's window verifier, to plain decoding outside it: a plain request builds no window, whose live state the runner
// would hold beside the prompt cache's copy (Bonsai-27B's plain request had peaked 181.6 MiB higher), and a drafted
// request on the same runner builds its windows.
func TestACellRunnerDecodesPlainlyOutsideWindows(t *testing.T) {
	// Serial: device fixtures share the one card.
	requireIntegration(t)
	cudatest.Require(t)
	family := qwen35Geometry{embedding: 64, ffn: 64, inner: 256, state: 128, groups: 1, rank: 2, gates: dtype.Q8_0}
	path := writeQwen35Fixture(t, family, "p", qwen35CellsContext, qwen35CellsWidth, qwen35CellsHeads, qwen35CellsKVHeads)
	runner := testcheck.Must(openF32FixtureRunner(path, OpenOptions{}))(t)
	t.Cleanup(func() { _ = runner.Close() })
	if !runner.windowsDecode() {
		t.Fatal("the hybrid over cells verifies outside Strata's window verifier")
	}
	greedy := testcheck.Must(sampling.New(sampling.Config{Temperature: 0}))(t)
	options := GenerateOptions{
		MaxNewTokens: 8, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: []tokenizer.TokenID{3, 5, 7, 4, 6}, PlainDecode: true,
		ContinueAfterEOG: true,
	}
	if _, _, err := runner.Generate(t.Context(), "", options); err != nil {
		t.Fatal(err)
	}
	if runner.qsaWindow != nil {
		t.Fatal("a plain request decoded in windows")
	}
	options.PlainDecode, options.SpeculativeDecode = false, true
	if _, _, err := runner.Generate(t.Context(), "", options); err != nil {
		t.Fatal(err)
	}
	if runner.qsaWindow == nil {
		t.Fatal("a drafted request verified outside windows")
	}
}
