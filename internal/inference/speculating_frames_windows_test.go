//go:build windows

package inference

import (
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// TestSpeculatingSessionReplaysItsFrames holds a NextN runner's speculation to replays of the graphs it captured.
//   - a runner no request has asked to speculate keeps the default captured graphs, and the first speculative request raises them
//   - the served speculative loop, its draft steps with its spans, captures nothing once two requests have warmed it, and keeps the plain greedy text
//   - seven verify spans from a fresh prompt accept and reject drafts in a fixed pattern, so spans of two and three tokens read pasts that spans of either length made, at both slot parities, across three cache pages; run a second time they capture nothing, keep every selection's and hidden row's bits, and verify the plain greedy text
func TestSpeculatingSessionReplaysItsFrames(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	prompt := []tokenizer.TokenID{1, 4, 5}
	for _, residency := range []recipe.ResidencyPolicy{recipe.ResidencyDeviceF32, recipe.ResidencyDeviceNative} {
		runner, err := openFixtureRunnerWithResidency(writeHermeticQwen35GGUF(t), OpenOptions{}, residency)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := runner.Close(); err != nil {
				t.Error(err)
			}
		})
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		generate := func(speculative bool, tokens int) ([]tokenizer.TokenID, uint64, uint64) {
			t.Helper()
			before, err := runner.cuda.Metrics(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
				MaxNewTokens: tokens, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: speculative,
				PromptTokenIDs: prompt, ContinueAfterEOG: true,
			})
			if err != nil {
				t.Fatalf("%s: %v", residency, err)
			}
			after, err := runner.cuda.Metrics(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			return ids[len(prompt):], after.GraphCaptures - before.GraphCaptures, after.GraphCacheCapacity
		}
		plain, _, kept := generate(false, 12)
		if runner.speculating || kept >= speculatingGraphExecs {
			t.Fatalf("%s: a runner never asked to speculate keeps %d graphs (speculating %v)", residency, kept, runner.speculating)
		}

		const served = 8
		for warm := range 2 {
			if ids, _, _ := generate(true, served); !slices.Equal(ids, plain[:served]) {
				t.Fatalf("%s: speculative request %d's text %v differs from the plain text %v", residency, warm, ids, plain)
			}
		}
		ids, captures, kept := generate(true, served)
		if !slices.Equal(ids, plain[:served]) {
			t.Fatalf("%s: the warm speculative text %v differs from the plain text %v", residency, ids, plain)
		}
		if kept != speculatingGraphExecs || captures != 0 {
			t.Fatalf("%s: a warm speculative request kept %d graphs and captured %d, want %d kept and its spans and draft steps to replay", residency, kept, captures, speculatingGraphExecs)
		}

		// accepted[r] says whether round r's one draft is the plain text's next token.
		accepted := []bool{true, false, true, true, false, true, false}
		type span struct {
			selected []tokenizer.TokenID
			hidden   []uint32
		}
		pass := func() ([]span, uint64, []tokenizer.TokenID) {
			t.Helper()
			runner.mu.Lock()
			defer runner.mu.Unlock()
			ctx := t.Context()
			boundary, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedy, prompt, nil)
			if err != nil {
				t.Fatal(err)
			}
			text := []tokenizer.TokenID{boundary.Selected}
			pending := []tokenizer.TokenID{boundary.Selected}
			before, err := runner.cuda.Metrics(ctx)
			if err != nil {
				t.Fatal(err)
			}
			var spans []span
			for round, keep := range accepted {
				next := plain[len(text)]
				draft := next
				if !keep {
					draft = (next + 1) % tokenizer.TokenID(runner.vocab.Len())
				}
				result, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedySpan, append(slices.Clone(pending), draft), boundary)
				if err != nil {
					t.Fatalf("%s round %d: %v", residency, round, err)
				}
				hidden := make([]uint32, len(result.SpanHidden.Data))
				for index, value := range result.SpanHidden.Data {
					hidden[index] = math.Float32bits(value)
				}
				spans = append(spans, span{selected: slices.Clone(result.SpanSelected), hidden: hidden})
				if keep {
					correction := result.SpanSelected[len(pending)]
					text = append(text, draft, correction)
					if err := boundary.Release(ctx); err != nil {
						t.Fatal(err)
					}
					boundary, pending = result, []tokenizer.TokenID{correction}
					continue
				}
				correction := result.SpanSelected[len(pending)-1]
				text = append(text, correction)
				if err := result.Release(ctx); err != nil {
					t.Fatal(err)
				}
				pending = append(pending, correction)
			}
			after, err := runner.cuda.Metrics(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if err := boundary.Release(ctx); err != nil {
				t.Fatal(err)
			}
			return spans, after.GraphCaptures - before.GraphCaptures, text
		}
		first, _, firstText := pass()
		second, captures, secondText := pass()
		if !slices.Equal(firstText, plain[:len(firstText)]) || !slices.Equal(secondText, firstText) {
			t.Fatalf("%s: the verified text %v / %v differs from the plain text %v", residency, firstText, secondText, plain)
		}
		for index := range first {
			if !slices.Equal(first[index].selected, second[index].selected) || !slices.Equal(first[index].hidden, second[index].hidden) {
				t.Fatalf("%s: span %d's selections or hidden rows differ between the passes", residency, index)
			}
		}
		if captures != 0 {
			t.Fatalf("%s: the second pass captured %d graphs, want every span to replay", residency, captures)
		}
	}
}
