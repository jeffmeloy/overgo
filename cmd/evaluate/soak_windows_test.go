package main

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/sampling"
)

// soakGenerations is how many sequential chat-template generations the
// soak drives through one runner: enough for the prompt lengths to cross
// every capacity band several times and for a program release to recur,
// the pattern that replayed a stale graph exec at generation 330 of the
// BBH chat pass (owner rule 2026-09-02: long-form inference reliability is
// as important as benchmarks).
const soakGenerations = 400

// soakAnswerTokens bounds each generation: the soak measures the prefill
// and session lifecycle across lengths, not long decodes.
const soakAnswerTokens = 8

// soakSentence is repeated to sweep prompt lengths; the repetition counts
// place prompts on both sides of every power-of-two capacity band up to
// the longest BBH prompt.
const soakSentence = "It is not always easy to see who is related to whom, and in which ways; the following argument pertains to this question. "

var soakRepeats = []int{1, 3, 7, 15, 31, 63, 2, 30, 62, 8, 4, 16}

// TestGenerationSoakSurvivesLengthSweep drives the smallest servable text
// model through hundreds of chat-template generations whose prompt lengths
// sweep the capacity bands, asserting that no generation errors and that
// the first prompt, repeated at the end, generates the same text as at the
// start. It runs under the device lane with the store resolved through the
// data-root contract.
func TestGenerationSoakSurvivesLengthSweep(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		t.Skip("data root is not resolvable: " + err.Error())
	}
	ctx := t.Context()
	store, err := overgodb.Open(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	path := soakModelPath(t, ctx, store)
	loaded, err := modelrecipe.ResolveActiveGGUF(ctx, store, path)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		_ = loaded.Close()
		t.Fatal(err)
	}
	defer runner.Close()
	runtime := chatShapedRuntime{runner}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	generate := func(prompt string) string {
		t.Helper()
		shaped, err := runtime.ShapeChatPrompt(prompt)
		if err != nil {
			t.Fatal(err)
		}
		// DeviceGreedy takes the device argmax path the evaluator's
		// generated-answer scoring uses, the one that reuses parameterized
		// decode sessions across generations.
		_, text, err := runtime.Generate(ctx, shaped, inference.GenerateOptions{
			MaxNewTokens: soakAnswerTokens, Sampler: greedy, DeviceGreedy: true,
		})
		if err != nil {
			t.Fatalf("generation failed: %v", err)
		}
		return text
	}
	first := generate(soakPrompt(1))
	// The page-boundary walk: for each capacity band, a past beyond the
	// band on the next page, then a past exactly at the band on its own
	// page, both appending into the same target page. A decode session
	// compiled over the larger page must not serve the smaller one (the
	// fault the 4B and 9B chat passes died of); the walk runs each pair in
	// both orders and with the exact shaped token counts.
	for _, band := range soakBands {
		// A prompt of exactly the band's token count prefills onto the
		// band's own page; its first decode step is the parameterized
		// append that met the larger past's cached session. On a model
		// whose page is small the stale session reads a neighbouring
		// allocation instead of faulting, so the walk also generates the
		// band prompt before the larger one and requires the same text
		// after it: greedy decoding over an intact cache is deterministic.
		onBand := soakPromptWithTokens(t, runtime, band)
		beyond := soakPromptWithTokens(t, runtime, band+soakBandOverhang)
		clean := generate(onBand)
		generate(beyond)
		t.Logf("boundary walk: %d then %d shaped tokens", band+soakBandOverhang, band)
		if after := generate(onBand); after != clean {
			t.Fatalf("band %d: the on-band prompt generated %q after a %d-token past, %q before it", band, after, band+soakBandOverhang, clean)
		}
		generate(beyond)
	}
	for index := 1; index < soakGenerations; index++ {
		repeats := soakRepeats[index%len(soakRepeats)]
		prompt := soakPrompt(repeats)
		if index%50 == 0 {
			t.Logf("generation %d (%d repeats, %d bytes)", index, repeats, len(prompt))
		}
		generate(prompt)
	}
	if again := generate(soakPrompt(1)); again != first {
		t.Fatalf("the first prompt generates %q after the soak, %q before it", again, first)
	}
}

// soakBands are the capacity bands the boundary walk straddles; the
// overhang puts the larger past of each pair on the next page while both
// pasts append into the same target page.
var soakBands = []int{256, 512}

const soakBandOverhang = 44

// soakFiller is one token per repetition on the models the soak runs, so
// the walk can hit an exact shaped token count.
const soakFiller = " and"

// soakPromptWithTokens builds a prompt whose shaped, special-token-parsed
// encoding has exactly the requested token count, growing a filler word
// by word; a tokenizer that cannot reach the count exactly fails the test
// rather than walking an approximate boundary.
func soakPromptWithTokens(t *testing.T, runtime chatShapedRuntime, tokens int) string {
	t.Helper()
	count := func(prompt string) int {
		shaped, err := runtime.ShapeChatPrompt(prompt)
		if err != nil {
			t.Fatal(err)
		}
		ids, err := runtime.TokenizeText(shaped, true, true)
		if err != nil {
			t.Fatal(err)
		}
		return len(ids)
	}
	base := "Q: Count the words."
	build := func(fillers int, tail string) string {
		return base + strings.Repeat(soakFiller, fillers) + tail + "\nA:"
	}
	// Grow by halves to just under the target, then one filler at a time.
	fillers := 0
	for have := count(build(fillers, "")); have < tokens; have = count(build(fillers, "")) {
		fillers += max(1, (tokens-have)/2)
	}
	for fillers > 0 && count(build(fillers, "")) > tokens {
		fillers--
	}
	if count(build(fillers, "")) == tokens {
		return build(fillers, "")
	}
	// One more filler overshoots, so a short tail word closes the gap.
	for _, tail := range []string{" a", " I", " so", " an", " and a"} {
		if count(build(fillers, tail)) == tokens {
			return build(fillers, tail)
		}
	}
	t.Fatalf("no filler count yields %d shaped tokens (%d fillers give %d)", tokens, fillers, count(build(fillers, "")))
	return ""
}

func soakPrompt(repeats int) string {
	return "Q: " + strings.Repeat(soakSentence, repeats) + "Is the argument valid or invalid?\nA:\nOptions:\nA. invalid\nB. valid\nAnswer with the letter only."
}

// soakModelPath picks the smallest present servable model whose declared
// evaluation domains admit text, the model every pass runs first.
func soakModelPath(t *testing.T, ctx context.Context, store *overgodb.Store) string {
	t.Helper()
	entries, err := discovery.ServableWithMemo(ctx, store, 256, nil)
	if err != nil {
		t.Fatal(err)
	}
	type sized struct {
		path  string
		bytes int64
	}
	var models []sized
	for _, entry := range entries {
		if !entry.Present || entry.Stale != "" || entry.Location == "" {
			continue
		}
		domains, declared, err := modelrecipe.EvalDomains(ctx, store, entry.Model)
		if err != nil {
			t.Fatal(err)
		}
		if declared && !slices.Contains(domains, evaluation.DomainText) {
			continue
		}
		info, err := os.Stat(entry.Location)
		if err != nil {
			continue
		}
		models = append(models, sized{path: entry.Location, bytes: info.Size()})
	}
	if len(models) == 0 {
		t.Skip("no servable text model is present")
	}
	slices.SortFunc(models, func(a, b sized) int { return cmp.Compare(a.bytes, b.bytes) })
	t.Log(fmt.Sprintf("soak model %s (%d bytes)", models[0].path, models[0].bytes))
	return models[0].path
}
