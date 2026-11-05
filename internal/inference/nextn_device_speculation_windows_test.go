//go:build windows

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

// TestNextNMTPDeviceSpeculationLossless proves the composed speculative
// loop on the hermetic hybrid: greedy generation with NextN MTP speculation
// emits token-for-token and byte-for-byte what plain greedy device decode
// emits, by default and, once the layer's drafts are refused, by request -
// speculation changes throughput, never output.
func TestNextNMTPDeviceSpeculationLossless(t *testing.T) {
	// Serial: device fixtures share the one card.
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openF32FixtureRunner(writeHermeticQwen35GGUF(t), OpenOptions{})
	testcheck.NoError(t, err)
	defer runner.Close()
	if !runner.SpeculatesByDefault() {
		t.Fatal("the model's own next-token layer does not speculate by default")
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	testcheck.NoError(t, err)
	drafted := 0
	byDefault := GenerateOptions{
		MaxNewTokens: 8, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: []tokenizer.TokenID{1, 4, 5},
		OnDraftSpan: func(drafts, _ int) { drafted += drafts },
	}
	if runner.deviceSpeculationReady(byDefault) {
		t.Fatal("a request that does not ask to speculate is admitted before Generate defaults it")
	}
	speculated, speculatedText, err := runner.Generate(t.Context(), "", byDefault)
	testcheck.NoError(t, err)
	if drafted == 0 {
		t.Fatal("a request that says nothing drafted no token")
	}

	// The oracle: the same generation decoded plainly.
	plainOptions := byDefault
	plainOptions.PlainDecode, plainOptions.OnDraftSpan = true, nil
	plain, plainText, err := runner.Generate(t.Context(), "", plainOptions)
	testcheck.NoError(t, err)
	if !slices.Equal(speculated, plain) || speculatedText != plainText {
		t.Fatalf("the default request emits %v %q, the plain one %v %q", speculated, speculatedText, plain, plainText)
	}

	// A refused layer no longer speculates by default, and a request that asks to still does.
	runner.ownDraftsRefused.Store(true)
	if runner.SpeculatesByDefault() {
		t.Fatal("a refused layer still speculates by default")
	}
	requested := byDefault
	requested.SpeculativeDecode = true
	if !runner.deviceSpeculationReady(requested) {
		t.Fatal("hybrid fixture does not admit NextN device speculation")
	}
	drafted = 0
	speculated, speculatedText, err = runner.Generate(t.Context(), "", requested)
	testcheck.NoError(t, err)
	if drafted == 0 {
		t.Fatal("a request that asked to speculate past the refusal drafted no token")
	}
	if !slices.Equal(speculated, plain) || speculatedText != plainText {
		t.Fatalf("the requested speculation emits %v %q, the plain one %v %q", speculated, speculatedText, plain, plainText)
	}
}
