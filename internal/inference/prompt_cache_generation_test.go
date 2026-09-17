package inference

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestPromptCacheReuseAfterDecodeTokensHost(t *testing.T) {
	var missing *Runner
	if missing.SupportsPromptCache() || new(Runner).SupportsPromptCache() {
		t.Fatal("uninitialized runner advertises prompt reuse")
	}
	runner := openPromptCacheHostRunner(t)
	if !runner.SupportsPromptCache() {
		t.Fatal("prepared runner does not advertise prompt reuse")
	}
	assertPromptCacheAfterDecode(t, runner)
}

func openPromptCacheHostRunner(t *testing.T) *Runner {
	t.Helper()
	path := testutil.HermeticLlamaGGUF(t, 16)
	runner, err := openFixtureRunnerWithResidency(path, OpenOptions{}, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runner.Close() })
	return runner
}

type promptCacheGeneration struct {
	ids        []tokenizer.TokenID
	logits     [][]float32
	evaluation PromptEvaluation
}

func generatePromptCacheFixture(t *testing.T, runner *Runner, prompt []tokenizer.TokenID, cached, cancel bool) promptCacheGeneration {
	t.Helper()
	ctx, stop := context.WithCancelCause(t.Context())
	defer stop(nil)
	sampler, err := sampling.New(sampling.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var result promptCacheGeneration
	result.ids, _, err = runner.Generate(ctx, "", GenerateOptions{
		PromptTokenIDs: prompt, MaxNewTokens: 4, ContinueAfterEOG: true,
		CachePrompt: cached, Sampler: sampler,
		OnPromptEvaluated: func(value PromptEvaluation) { result.evaluation = value },
		OnToken: func(event TokenEvent) error {
			result.logits = append(result.logits, slices.Clone(event.Logits))
			if cancel {
				stop(context.Canceled)
				return context.Cause(ctx)
			}
			return nil
		},
	})
	if cancel {
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel error=%v", err)
		}
	} else if err != nil {
		t.Fatal(err)
	}
	return result
}

func assertPromptCacheAfterDecode(t *testing.T, runner *Runner) {
	t.Helper()
	prompt := []tokenizer.TokenID{1, 4, 5}
	for _, capacity := range []int{1, 2} {
		runner.promptCacheCapacity = capacity
		for _, next := range [][]tokenizer.TokenID{prompt, {1, 4, 5, 6}, {1, 4, 7}, {1, 4}} {
			for _, cancelled := range []bool{false, true} {
				if err := runner.ClearPromptCaches(t.Context()); err != nil {
					t.Fatal(err)
				}
				cold := generatePromptCacheFixture(t, runner, next, false, false)
				generatePromptCacheFixture(t, runner, prompt, true, cancelled)
				warm := generatePromptCacheFixture(t, runner, next, true, false)
				if !slices.Equal(cold.ids, warm.ids) {
					t.Fatalf("capacity=%d prompt=%v cancel=%t cold=%v reused=%v", capacity, next, cancelled, cold.ids, warm.ids)
				}
				if warm.evaluation.Tokens != len(next) || cold.evaluation.Cached != 0 {
					t.Fatalf("accounting cold=%+v warm=%+v", cold.evaluation, warm.evaluation)
				}
				if slices.Equal(next, prompt) && warm.evaluation.Cached != len(prompt) {
					t.Fatalf("exact prompt not reused: %+v", warm.evaluation)
				}
				if len(warm.logits) != 4 || len(cold.logits) != 4 {
					t.Fatal("incomplete token/logit comparison")
				}
				for token := range cold.logits {
					if len(cold.logits[token]) == 0 || len(warm.logits[token]) != len(cold.logits[token]) {
						t.Fatal("missing logits")
					}
					for index, want := range cold.logits[token] {
						// Same execution backend, independently cold prefill. The
						// existing hermetic device parity contract allows 2e-4.
						if delta := math.Abs(float64(warm.logits[token][index] - want)); math.IsNaN(delta) || delta > 2e-4 {
							t.Fatalf("prompt=%v token=%d logit=%d delta=%g", next, token, index, delta)
						}
					}
				}
				if len(runner.promptCaches) > capacity {
					t.Fatal("cache exceeded capacity")
				}
			}
		}
	}
	if err := runner.ClearPromptCaches(t.Context()); err != nil {
		t.Fatal(err)
	}
	if len(runner.promptCaches) != 0 {
		t.Fatal("cache survived explicit clear")
	}
}

func TestPromptCacheTotalTurnCostHost(t *testing.T) {
	// Keep cost observations separate from the CUDA/host numerical acceptance.
	// A matched ABBA observation includes prefill, four decoded tokens,
	// callbacks and text decoding. It describes this tiny fixture only; no
	// statistical or production-model speed claim follows from these samples.
	runner := openPromptCacheHostRunner(t)
	prompt := []tokenizer.TokenID{1, 4, 5}
	runner.promptCacheCapacity = 1
	generatePromptCacheFixture(t, runner, prompt, false, false) // warm executor
	generatePromptCacheFixture(t, runner, prompt, true, false)  // establish cache
	var oracle []tokenizer.TokenID
	for index, cached := range []bool{false, true, true, false} {
		started, err := processmeasure.Counter()
		if err != nil {
			t.Fatal(err)
		}
		observed := generatePromptCacheFixture(t, runner, prompt, cached, false)
		finished, err := processmeasure.Counter()
		if err != nil {
			t.Fatal(err)
		}
		if index == 0 {
			oracle = observed.ids
		}
		if !slices.Equal(observed.ids, oracle) {
			t.Fatal("paired timing changed output")
		}
		t.Logf("paired host total-turn sample=%d cache=%t prompt=%v output=%v cached_tokens=%d wall=%s", index, cached, prompt, observed.ids[len(prompt):], observed.evaluation.Cached, finished-started)
	}
}
