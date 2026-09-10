package inference

import (
	"testing"

	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestPromptMeasurementClock(t *testing.T) {
	path := testutil.HermeticLlamaGGUF(t, 16)
	runner, err := openFixtureRunnerWithResidency(path, OpenOptions{}, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	sampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	for _, cached := range []bool{false, true} {
		var measured PromptEvaluation
		calls := 0
		before, err := processmeasure.Counter()
		if err != nil {
			t.Fatal(err)
		}
		ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{PromptTokenIDs: []tokenizer.TokenID{1, 4}, MaxNewTokens: 2, ContinueAfterEOG: true, CachePrompt: cached, Sampler: sampler, OnPromptEvaluated: func(value PromptEvaluation) { measured = value; calls++ }})
		after, clockErr := processmeasure.Counter()
		if err != nil || clockErr != nil {
			t.Fatalf("generation: %v; clock: %v", err, clockErr)
		}
		if calls != 1 || measured.Tokens != 2 || measured.Duration <= 0 || measured.Duration > after-before || len(ids) < 2 {
			t.Fatalf("prompt measurement %+v calls=%d generated=%v wall=%s", measured, calls, ids, after-before)
		}
	}
}
