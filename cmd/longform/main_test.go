package main

import (
	"strings"
	"testing"

	"overgo/internal/longform"
)

// The tool takes either the whole servable text catalog or named
// models, never both and never neither, and its report reads the
// measure, the verdict, the prompt tail, and the head of the output.
func TestOptionsAndReport(t *testing.T) {
	if _, err := parseOptions(nil); err == nil {
		t.Fatal("no target accepted")
	}
	if _, err := parseOptions([]string{"-all", "model.gguf"}); err == nil {
		t.Fatal("-all with a named model accepted")
	}
	options, err := parseOptions([]string{"-publish", "-show", "12", "a.gguf", "b.gguf"})
	if err != nil || !options.Publish || options.All || options.OutputPrefix != 12 || len(options.Models) != 2 ||
		options.PromptFile != defaultPromptFile || options.Repository != "overgodb-store" {
		t.Fatalf("options = %+v, %v", options, err)
	}
	floors := longform.DeclaredFloors()
	result := longform.Result{
		Measure: longform.Measure{
			PromptTokens: 2000, OutputTokens: 256, PromptTokensPerSecond: 2313.8, DecodeTokensPerSecond: 73.0,
			DistinctFourGramRatio: 0.186, LongestRepeatedSpan: 134,
			Score: longform.ContextScore{ScoreTokens: 128, LongContextNLL: 2.955, ShortContextNLL: 3.186, ContextGain: 0.231},
		},
		Short: longform.ShortRates{PromptTokensPerSecond: 813.6, DecodeTokensPerSecond: 75.9}, Floors: floors,
		PromptTail: "### Model and artifact lifecycle\n", Output: "- GGUF and safetensors intake, conversion, quantization,\n model construction",
	}
	result.Verdict = longform.Judge(result.Measure, result.Short, floors)
	var output strings.Builder
	report(&output, result, 12)
	text := output.String()
	for _, want := range []string{
		"prompt 2000 tok at 2313.8 tok/s (short-prompt 813.6)", "decode 256 tok at 73.0 tok/s (short-prompt 75.9)",
		"distinct 4-gram ratio 0.186, longest repeated span 134 tok", "NLL 2.955 long / 3.186 short", "-> PASS",
		"prompt tail: ...### Model and artifact lifecycle ", "output: - GGUF and s...",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("report lacks %q:\n%s", want, text)
		}
	}
}
