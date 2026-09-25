package runrecord_test

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/benchmarkrecord"
	"overgo/internal/recipe"
)

// TestBenchmarkComparisonRecord keeps the run-record consumer contract wired
// to the shared, content-addressed comparison document.
func TestBenchmarkComparisonRecord(t *testing.T) {
	model, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	recipeID, _ := artifact.IdentifyBytes(artifact.KindRecipe, []byte("recipe"))
	environment, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("environment"))
	prompt := strings.Repeat("a", 64)
	token := strings.Repeat("b", 64)
	baseline := benchmarkrecord.BenchmarkConfiguration{Tokens: 2, Runs: 1}
	candidate := baseline
	candidate.CachePrompt = true
	observation := func(milliseconds float64) benchmarkrecord.BenchmarkObservation {
		return benchmarkrecord.BenchmarkObservation{
			PromptDigest: prompt, TokenDigests: []string{token},
			PromptTokens: 2, OutputTokens: 2, HostLogitTokens: 2,
			TotalMilliseconds: milliseconds, TTFTMilliseconds: 1,
		}
	}
	record, err := benchmarkrecord.NewBenchmarkComparisonRecord(benchmarkrecord.BenchmarkComparisonRecord{
		Identity: benchmarkrecord.BenchmarkIdentity{
			Model: model, Recipe: recipeID, TokenizerContainer: model,
			Environment: environment, CodeCommit: strings.Repeat("c", 40),
			ModuleDigest: strings.Repeat("d", 64), WorkloadDigest: strings.Repeat("e", 64),
			PromptDigests: []string{prompt}, RealizedResidency: recipe.RealizedDeviceNative,
		},
		Factor: "cache_prompt", Baseline: baseline, Candidate: candidate,
		Exclusive: true, CacheUpperBound: true,
		Pairs: []benchmarkrecord.BenchmarkPair{{
			Index: 0, First: "baseline",
			Baseline: observation(10), Candidate: observation(9),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := record.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	content, err := record.Content()
	if err != nil || len(content.Data) == 0 {
		t.Fatalf("shared record content = %+v, %v", content, err)
	}
	if _, err := record.Batch("benchmark/comparison"); err != nil {
		t.Fatal(err)
	}
	if record.Outcome != "all-pairs-faster" || record.MedianDeltaMilliseconds != -1 {
		t.Fatalf("shared comparison verdict = %+v", record)
	}
}
