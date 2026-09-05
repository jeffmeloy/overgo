package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/longform"
)

// The tool takes either the whole servable text catalog or named
// models, never both and never neither, a check does not publish, and
// its report reads the short shape, every rung, the judged verdict, the
// prompt tail, and the head of the output.
func TestOptionsAndReport(t *testing.T) {
	t.Setenv(dataroot.Env, "")
	if _, err := parseOptions(nil); err == nil {
		t.Fatal("no target accepted")
	}
	if _, err := parseOptions([]string{"-all", "model.gguf"}); err == nil {
		t.Fatal("-all with a named model accepted")
	}
	if _, err := parseOptions([]string{"-all", "-check", "-publish"}); err == nil {
		t.Fatal("-check with -publish accepted")
	}
	options, err := parseOptions([]string{"-publish", "-budget", "1m", "-show", "12", "a.gguf", "b.gguf"})
	if err != nil || !options.Publish || options.All || options.Check || options.OutputPrefix != 12 || len(options.Models) != 2 ||
		options.Repository != "overgodb-store" {
		t.Fatalf("options = %+v, %v", options, err)
	}
	floors := longform.DeclaredFloors()
	judged := longform.Measure{
		PromptTokens: 2048, OutputTokens: 256, PromptTokensPerSecond: 2313.8, DecodeTokensPerSecond: 73.0,
		DistinctFourGramRatio: 0.186, LongestRepeatedSpan: 134,
		Score: longform.ContextScore{ScoreTokens: 128, LongContextNLL: 2.955, ShortContextNLL: 3.186, ContextGain: 0.231},
	}
	result := longform.Result{
		Measure: judged,
		Short:   longform.ShortRates{PromptTokensPerSecond: 813.6, DecodeTokensPerSecond: 75.9}, Floors: floors,
		Shape:      longform.ShortShape{PromptTokens: 160, OutputIDs: make([]int32, 64), NLL: 3.4, Measure: longform.Measure{DecodeTokensPerSecond: 80}},
		Rungs:      []longform.Rung{{Measure: longform.Measure{PromptTokens: 1024, OutputTokens: 256, PromptTokensPerSecond: 2000, DecodeTokensPerSecond: 74}}, {Measure: judged}},
		LadderStop: "rung 4096 prefill took 70.0s, past the 60s rung budget",
		PromptTail: "### Model and artifact lifecycle\n", Output: "- GGUF and safetensors intake, conversion, quantization,\n model construction",
	}
	result.Verdict = longform.Judge(result.Measure, result.Short, floors)
	var output strings.Builder
	report(&output, result, 12)
	text := output.String()
	for _, want := range []string{
		"short 160->64 tok: NLL 3.400, decode 80.0 tok/s",
		"rung 1024: prompt 2000.0 tok/s, decode 256 tok at 74.0 tok/s",
		"rung 2048: prompt 2313.8 tok/s", "NLL 2.955 long / 3.186 short (gain 0.231)",
		"ladder stopped: rung 4096 prefill took 70.0s",
		"judged rung 2048: prompt 2313.8 tok/s (short-prompt 813.6); decode 256 tok at 73.0 tok/s (short-prompt 75.9) -> PASS",
		"prompt tail: ...### Model and artifact lifecycle ", "output: - GGUF and s...",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("report lacks %q:\n%s", want, text)
		}
	}
}

func TestLongformRepositoryResolution(t *testing.T) {
	root := t.TempDir()
	t.Setenv(dataroot.Env, root)
	parse := func(args ...string) (options, error) {
		return parseOptions(append(args, "-budget", "1m", "model.gguf"))
	}
	resolved, err := parse()
	if err != nil || resolved.Repository != filepath.Join(root, "overgodb-store") {
		t.Fatalf("canonical data root: %+v, %v", resolved, err)
	}
	t.Setenv(dataroot.Env, filepath.Join(root, "missing"))
	if _, err := parse(); err == nil {
		t.Fatal("invalid canonical data root accepted")
	}
	if explicit, err := parse("-repo", root); err != nil || explicit.Repository != root {
		t.Fatalf("explicit repository lost authority: %+v, %v", explicit, err)
	}
	t.Setenv(dataroot.Env, "")
	if err := os.WriteFile(filepath.Join(root, dataroot.ConfigFile), []byte(`{"store":"local-store"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if local, err := parse("-root", root); err != nil || local.Repository != filepath.Join(root, "local-store") {
		t.Fatalf("repository-local data root: %+v, %v", local, err)
	}
}
