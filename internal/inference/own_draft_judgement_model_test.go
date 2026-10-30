//go:build modeltest

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modeltest"
	"overgo/internal/sampling"
)

// TestAServedModelDraftsOnlyWhereDraftingIsFaster: a runner opened as a served model is (OpenOptions.JudgeOwnDrafts)
// times a plain and a drafted decoding through its model's own next-token layer at open, and drafts by default only
// where drafting is faster. Qwen3.5-4B, whose four-token verify costs about two plain steps, decodes plainly, and a
// request that asks to speculate still drafts, the plain decode's tokens; Qwen3.8-27B, whose window costs about a plain
// step, drafts by default.
func TestAServedModelDraftsOnlyWhereDraftingIsFaster(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		drafts bool
	}{
		{"Qwen3.5-4B", false},
		{"Qwen3.8-27B-UD-Q6_K.gguf", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			runner, err := openNativeFixtureRunner(modeltest.Directory(t, test.name), OpenOptions{JudgeOwnDrafts: true})
			if err != nil {
				t.Fatal(err)
			}
			defer runner.Close()
			if runner.SpeculatesByDefault() != test.drafts {
				t.Fatalf("judged at open, %s drafts by default %t, want %t", test.name, runner.SpeculatesByDefault(), test.drafts)
			}
			drafted := 0
			asked := GenerateOptions{
				MaxNewTokens: 32, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: true,
				OnDraftSpan: func(drafts, _ int) { drafted += drafts },
			}
			const prompt = "Explain, step by step, why the sum of the first n odd numbers is n squared."
			speculated, _, err := runner.Generate(t.Context(), prompt, asked)
			if err != nil {
				t.Fatal(err)
			}
			plain := asked
			plain.SpeculativeDecode, plain.PlainDecode, plain.OnDraftSpan = false, true, nil
			decoded, _, err := runner.Generate(t.Context(), prompt, plain)
			if err != nil {
				t.Fatal(err)
			}
			if drafted == 0 || !slices.Equal(speculated, decoded) {
				t.Fatalf("a request that asks to speculate drafted %d tokens and emits %v, the plain decode %v", drafted, speculated, decoded)
			}
		})
	}
}
