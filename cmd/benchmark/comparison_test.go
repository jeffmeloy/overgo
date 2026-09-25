package main

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/tokenizer"
)

func TestBenchmarkResultTokenDigest(t *testing.T) {
	digest := newTokenDigest()
	for _, id := range []int32{1, -2, 7} {
		digest.add(tokenizer.TokenID(id))
	}
	expected := append([]byte(tokenDigestDomain),
		1, 0, 0, 0, 254, 255, 255, 255, 7, 0, 0, 0)
	sum := sha256.Sum256(expected)
	if got, want := digest.sum(), hex.EncodeToString(sum[:]); got != want || digest.count != 3 {
		t.Fatalf("token digest = %s count=%d, want %s count=3", got, digest.count, want)
	}
	reversed := newTokenDigest()
	for _, id := range []int32{7, -2, 1} {
		reversed.add(tokenizer.TokenID(id))
	}
	if reversed.sum() == digest.sum() {
		t.Fatal("token order did not change digest")
	}
	first := newTokenDigest()
	first.add(1)
	second := newTokenDigest()
	second.add(2)
	if first.sum() == second.sum() {
		t.Fatal("sequence digests collapsed")
	}
}

func TestPromptSuiteRotation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "suite.json")
	if err := os.WriteFile(path, []byte(`["cold","repeat","rotate"]`), 0o644); err != nil {
		t.Fatal(err)
	}
	opts, err := parseOptions([]string{"-prompt-suite", path, "-runs", "5", "model.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	prompts, err := benchmarkPrompts(opts)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"cold", "repeat", "rotate", "cold", "repeat"}
	for index, expected := range want {
		if got := prompts[benchmarkPromptIndex(index, len(prompts))]; got != expected {
			t.Fatalf("run %d prompt = %q, want %q", index, got, expected)
		}
	}
	if benchmarkPromptIndex(-1, len(prompts)) != 0 {
		t.Fatal("warmup did not use the first prompt")
	}
	if benchmarkWorkloadDigest(prompts) == benchmarkWorkloadDigest([]string{"rotate", "repeat", "cold"}) {
		t.Fatal("workload identity ignores prompt order")
	}
	for _, invalid := range []string{`[]`, `["ok",""]`, `{"prompt":"wrong type"}`} {
		if err := os.WriteFile(path, []byte(invalid), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := benchmarkPrompts(opts); err == nil {
			t.Fatalf("accepted invalid suite %s", invalid)
		}
	}
}

func TestPairedComparison(t *testing.T) {
	baseline, candidate := pairedFixture(t)
	first := []string{"baseline", "candidate", "baseline"}
	comparison, err := compareBenchmarkResults(baseline, candidate, "cache_prompt", first)
	if err != nil {
		t.Fatal(err)
	}
	if !comparison.ID.Valid() || comparison.Outcome != "all-pairs-faster" || comparison.MedianDeltaMilliseconds != -10 || comparison.CacheUpperBound {
		t.Fatalf("paired comparison = %+v", comparison)
	}
	if len(comparison.Pairs) != 3 || len(comparison.Warmups) != 2 || comparison.Pairs[1].First != "candidate" {
		t.Fatalf("raw paired observations absent: %+v", comparison)
	}
	for name, change := range map[string]func(*benchmarkResult){
		"model":     func(r *benchmarkResult) { r.ModelID = baseline.RecipeID },
		"residency": func(r *benchmarkResult) { r.RealizedResidency = recipe.RealizedHostCache },
		"output":    func(r *benchmarkResult) { r.Runs[1].OutputDigests[0] = benchmarkPromptDigest("wrong token") },
		"workload":  func(r *benchmarkResult) { r.PromptSuite[1] = "different" },
	} {
		t.Run(name, func(t *testing.T) {
			changed := candidate
			changed.Runs = slices.Clone(candidate.Runs)
			for index := range changed.Runs {
				changed.Runs[index].OutputDigests = slices.Clone(changed.Runs[index].OutputDigests)
			}
			changed.PromptSuite = slices.Clone(candidate.PromptSuite)
			change(&changed)
			if _, err := compareBenchmarkResults(baseline, changed, "cache_prompt", first); err == nil {
				t.Fatal("incomparable result accepted")
			}
		})
	}
	if _, err := compareBenchmarkResults(baseline, candidate, "cache_prompt", []string{"candidate", "baseline", "candidate"}); err == nil || !strings.Contains(err.Error(), "pair") {
		t.Fatalf("non-interleaved order accepted: %v", err)
	}
}

func pairedFixture(t *testing.T) (benchmarkResult, benchmarkResult) {
	t.Helper()
	prompts := []string{"first", "second"}
	baseline := benchmarkResult{
		Residency: recipe.ResidencyDeviceNative, RealizedResidency: recipe.RealizedDeviceNative,
		EndOfSequenceIgnored: true, SamplingProtocol: greedyBudgetProtocol,
		Prompt: prompts[0], PromptSuite: prompts, WorkloadDigest: benchmarkWorkloadDigest(prompts),
		TokensPerSequence: 4, RequestedRuns: 3, RequestedWarmup: 1,
		WarmupRuns: []runMetrics{{Run: -1, PromptDigest: benchmarkPromptDigest(prompts[0]), OutputDigests: []string{fixtureOutputDigest(4)},
			PromptTokens: 2, OutputTokens: 4, HostLogitTokens: 4, TotalMilliseconds: 40}},
	}
	baseline.Runs = make([]runMetrics, 3)
	for index := range baseline.Runs {
		baseline.Runs[index] = runMetrics{Run: index, PromptDigest: benchmarkPromptDigest(prompts[index%len(prompts)]),
			OutputDigests: []string{fixtureOutputDigest(4)}, PromptTokens: 2, OutputTokens: 4,
			HostLogitTokens: 4, TotalMilliseconds: float64(100 + 10*index), TTFTMilliseconds: 10}
	}
	bindFixtureBenchmarkIdentity(t, &baseline, "0123456789abcdef0123456789abcdef01234567")
	candidate := baseline
	candidate.PromptSuite = slices.Clone(baseline.PromptSuite)
	candidate.WarmupRuns = slices.Clone(baseline.WarmupRuns)
	candidate.Runs = slices.Clone(baseline.Runs)
	candidate.CachePrompt = true
	for index := range candidate.Runs {
		candidate.Runs[index].TotalMilliseconds -= 10
	}
	var err error
	candidate.OptionsDigest, err = benchmarkOptionsDigest(options{Tokens: 4, Runs: 3, Warmup: 1, CachePrompt: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return baseline, candidate
}
