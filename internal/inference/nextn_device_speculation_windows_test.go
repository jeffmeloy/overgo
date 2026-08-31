//go:build windows

package inference

import (
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// TestNextNMTPDeviceSpeculationLossless proves the composed speculative
// loop on the hermetic hybrid: greedy generation with NextN MTP speculation
// emits token-for-token what plain greedy device decode emits — speculation
// changes throughput, never output.
func TestNextNMTPDeviceSpeculationLossless(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := writeHermeticQwen35GGUF(t)
	runner, err := openF32FixtureRunner(path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	prompt := []tokenizer.TokenID{1, 4, 5}
	options := GenerateOptions{
		MaxNewTokens: 8, Sampler: greedy, DeviceGreedy: true,
		SpeculativeDecode: true, PromptTokenIDs: prompt,
	}
	if !runner.deviceSpeculationReady(options) {
		t.Fatal("hybrid fixture does not admit NextN device speculation")
	}
	speculated, speculatedText, err := runner.Generate(context.Background(), "", options)
	if err != nil {
		t.Fatal(err)
	}

	// Plain-loop oracle: the same generation without the opt-in.
	plainOptions := options
	plainOptions.SpeculativeDecode = false
	if runner.deviceSpeculationReady(plainOptions) {
		t.Fatal("plain generation unexpectedly admits speculation")
	}
	plain, plainText, err := runner.Generate(context.Background(), "", plainOptions)
	if err != nil {
		t.Fatal(err)
	}
	if len(speculated) != len(plain) {
		t.Fatalf("token counts differ: speculative=%v plain=%v", speculated, plain)
	}
	for index := range plain {
		if speculated[index] != plain[index] {
			t.Fatalf(
				"speculative output diverged at %d: %v vs %v",
				index, speculated, plain,
			)
		}
	}
	if speculatedText != plainText {
		t.Fatalf("decoded text differs: %q vs %q", speculatedText, plainText)
	}
}
