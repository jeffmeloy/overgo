//go:build windows

package inference

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// TestCardTopKDrawsAsTheHostSampler holds a sampled request's draws from the card's top K of each row
// (GenerateOptions.DeviceTopK) to the host sampler's draws from the whole row: on the hermetic Flash-Next with its
// draft layer, which steps and verifies its spans, and on the fixture at Strata's geometry, which decodes through
// windows, a request at temperature one with the top-k and top-p cuts decodes the same text from the same seed either
// way, plainly and speculating, and draws from the top K with no row's logits brought to the host.
func TestCardTopKDrawsAsTheHostSampler(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const seeds, generated = 8, 24
	prompt := []tokenizer.TokenID{4, 9, 17}
	hermetic := writeHermeticFlashNextGGUF(t)
	windows := writeFlashNextFixture(t, strataFlashNextWindows)
	for _, fixture := range []struct {
		name, trunk, draft string
		windows            bool
	}{
		{"hermetic", hermetic, writeHermeticFlashNextMTPGGUF(t, hermetic, false), false},
		{"windows", windows, writeFlashNextMTPFixture(t, windows, false, strataFlashNextWindows), true},
	} {
		runner, err := openFixtureRunnerWithResidency(fixture.trunk, OpenOptions{Draft: fixture.draft}, recipe.ResidencyDeviceNative)
		if err != nil {
			t.Fatal(err)
		}
		if runner.windowsDecode() != fixture.windows {
			_ = runner.Close()
			t.Fatalf("%s: the runner decodes through windows %t", fixture.name, runner.windowsDecode())
		}
		drafted := 0
		for seed := range int64(seeds) {
			generate := func(plain, card bool) []tokenizer.TokenID {
				t.Helper()
				sampler, err := sampling.New(sampling.Config{Temperature: 1, TopK: 20, TopP: 0.95, Seed: seed + 1})
				if err != nil {
					t.Fatal(err)
				}
				rows := 0
				options := GenerateOptions{
					MaxNewTokens: generated, Sampler: sampler, PromptTokenIDs: prompt, ContinueAfterEOG: true,
					PlainDecode: plain, DeviceTopK: card,
					OnToken: func(event TokenEvent) error {
						if event.Logits != nil {
							rows++
						}
						return nil
					},
				}
				if !plain {
					options.OnDraftSpan = func(count, _ int) { drafted += count }
				}
				ids, _, err := runner.Generate(t.Context(), "", options)
				if err != nil {
					_ = runner.Close()
					t.Fatal(err)
				}
				if card == (rows != 0) {
					_ = runner.Close()
					t.Fatalf("%s seed %d plain=%t top-K=%t: %d of %d tokens drawn from a row brought to the host", fixture.name, seed, plain, card, rows, generated)
				}
				return ids[len(prompt):]
			}
			for _, plain := range []bool{true, false} {
				if whole, card := generate(plain, false), generate(plain, true); !slices.Equal(card, whole) {
					_ = runner.Close()
					t.Fatalf("%s seed %d plain=%t: the card's top K drew %v, the whole rows %v", fixture.name, seed, plain, card, whole)
				}
			}
		}
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
		if drafted == 0 {
			t.Fatalf("%s: no window drafted", fixture.name)
		}
		t.Logf("%s: %d seeds draw the same text from the top K as from the whole rows, plainly and over %d drafts", fixture.name, seeds, drafted)
	}
}

// TestSampledToolRequestDrawsFromTheTopKUntilItsTrigger: a sampled tool request's lazy grammar masks no token until its
// trigger, so its rows come down as the card's top K until a token it accepts triggers it, and whole after, which the
// grammar masks: on the stepping fixture and the windows' fixture, plainly and speculating, a seed's text is the whole
// rows' text, its tokens through the trigger drawn from the top K and those after from whole rows; plainly, the text
// through the trigger is the seed's text without the grammar, and the grammar's token follows it.
func TestSampledToolRequestDrawsFromTheTopKUntilItsTrigger(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const generated = 24
	prompt := []tokenizer.TokenID{4, 9, 17}
	hermetic := writeHermeticFlashNextGGUF(t)
	windows := writeFlashNextFixture(t, strataFlashNextWindows)
	for _, fixture := range []struct{ name, trunk, draft string }{
		{"hermetic", hermetic, writeHermeticFlashNextMTPGGUF(t, hermetic, false)},
		{"windows", windows, writeFlashNextMTPFixture(t, windows, false, strataFlashNextWindows)},
	} {
		runner, err := openFixtureRunnerWithResidency(fixture.trunk, OpenOptions{Draft: fixture.draft}, recipe.ResidencyDeviceNative)
		if err != nil {
			t.Fatal(err)
		}
		// generate decodes the seed's text under the grammar, if any, which may end it, and counts the tokens drawn from
		// whole rows.
		generate := func(grammar string, trigger tokenizer.TokenID, plain, card bool) ([]tokenizer.TokenID, int) {
			t.Helper()
			config := sampling.Config{Temperature: 1, TopK: 20, TopP: 0.95, Seed: 3}
			if grammar != "" {
				compiled, err := runner.CompileLazyGBNF(grammar, "", nil, []tokenizer.TokenID{trigger})
				if err != nil {
					t.Fatal(err)
				}
				config.GBNF = compiled
			}
			sampler, err := sampling.New(config)
			if err != nil {
				t.Fatal(err)
			}
			rows := 0
			ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
				MaxNewTokens: generated, Sampler: sampler, PromptTokenIDs: prompt, ContinueAfterEOG: grammar == "",
				PlainDecode: plain, DeviceTopK: card,
				OnToken: func(event TokenEvent) error {
					if event.Logits != nil {
						rows++
					}
					return nil
				},
			})
			if err != nil {
				_ = runner.Close()
				t.Fatal(err)
			}
			return ids[len(prompt):], rows
		}
		free, _ := generate("", 0, true, false)
		// The trigger is a token of the free text at its first place, after a few drawn before it.
		at := 4
		for slices.Index(free, free[at]) != at {
			at++
		}
		// A token of the free text's own, neither the trigger, the token after it nor one ending generation, which the
		// grammar repeats.
		trigger := free[at]
		forced := free[slices.IndexFunc(free, func(id tokenizer.TokenID) bool {
			return id != trigger && id != free[at+1] && !runner.vocab.IsEOG(id)
		})]
		grammar := fmt.Sprintf("root ::= <[%d]> <[%d]>+", trigger, forced)
		for _, plain := range []bool{true, false} {
			whole, wholeRows := generate(grammar, trigger, plain, false)
			card, cardRows := generate(grammar, trigger, plain, true)
			if !slices.Equal(card, whole) || wholeRows != len(whole) {
				_ = runner.Close()
				t.Fatalf("%s plain=%t: the top K drew %v, the whole rows %v (%d of them whole)", fixture.name, plain, card, whole, wholeRows)
			}
			triggered := slices.Index(card, trigger)
			if triggered < 0 || cardRows != len(card)-triggered-1 || triggered+1 == len(card) || card[triggered+1] != forced {
				_ = runner.Close()
				t.Fatalf("%s plain=%t: %v, the trigger %d at %d, %d tokens drawn from whole rows", fixture.name, plain, card, trigger, triggered, cardRows)
			}
			if plain && !slices.Equal(card[:triggered+1], free[:at+1]) {
				_ = runner.Close()
				t.Fatalf("%s: the text through the trigger %v is not the free text's %v", fixture.name, card[:triggered+1], free[:at+1])
			}
		}
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: the trigger %d at %d of the free text %v", fixture.name, trigger, at, free)
	}
}

// TestSampledToolRequestTriggersOnAnAcceptedDraft: with a draft layer that copies the trunk's selection, a sampled
// request's draft is kept where its text repeats a token, and a trigger pattern of that token twice completes on the
// kept draft, inside a window whose later rows came down as their top K: the window ends there, and the tokens after
// it come from whole rows, which the grammar masks, the text the whole rows' from the same seed, on the stepping
// fixture and the windows' fixture.
func TestSampledToolRequestTriggersOnAnAcceptedDraft(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const generated, seeds = 24, 128
	prompt := []tokenizer.TokenID{4, 9, 17}
	hermetic := writeHermeticFlashNextGGUF(t)
	windows := writeFlashNextFixture(t, strataFlashNextWindows)
	for _, fixture := range []struct{ name, trunk, draft string }{
		{"hermetic", hermetic, writeHermeticFlashNextMTPGGUF(t, hermetic, true)},
		{"windows", windows, writeFlashNextMTPFixture(t, windows, true, strataFlashNextWindows)},
	} {
		runner, err := openFixtureRunnerWithResidency(fixture.trunk, OpenOptions{Draft: fixture.draft}, recipe.ResidencyDeviceNative)
		if err != nil {
			t.Fatal(err)
		}
		text := func(ids []tokenizer.TokenID) string {
			var joined strings.Builder
			for _, id := range ids {
				piece, err := runner.vocab.DecodePiece(id, false)
				if err != nil {
					t.Fatal(err)
				}
				joined.WriteString(piece)
			}
			return joined.String()
		}
		// generate speculates the seed's text under the grammar, if any, and answers the places of its kept drafts, a
		// window's first tokens (each window reports its drafts once it has drawn them), and its tokens drawn from whole
		// rows.
		generate := func(seed int64, grammar *sampling.GBNFGrammar, card bool) ([]tokenizer.TokenID, map[int]bool, int) {
			t.Helper()
			sampler, err := sampling.New(sampling.Config{Temperature: 1, TopK: 20, TopP: 0.95, Seed: seed, GBNF: grammar})
			if err != nil {
				t.Fatal(err)
			}
			rows, emitted, kept := 0, 0, map[int]bool{}
			ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
				MaxNewTokens: generated, Sampler: sampler, PromptTokenIDs: prompt, DeviceTopK: card,
				OnDraftSpan: func(_, accepted int) {
					for place := emitted - accepted - 1; place < emitted-1; place++ {
						kept[place] = true
					}
				},
				OnToken: func(event TokenEvent) error {
					if emitted++; event.Logits != nil {
						rows++
					}
					return nil
				},
			})
			if err != nil {
				_ = runner.Close()
				t.Fatalf("%s seed %d: %v", fixture.name, seed, err)
			}
			return ids[len(prompt):], kept, rows
		}
		triggered := 0
		for seed := range int64(seeds) {
			free, kept, _ := generate(seed, nil, true)
			for at := 1; at < len(free)-1; at++ {
				repeated := free[at]
				pattern := text([]tokenizer.TokenID{repeated, repeated})
				// The repeat is a kept draft and the pattern's first match, ending at the repeated token.
				if !kept[at] || free[at-1] != repeated || strings.Index(text(free[:at+1]), pattern) != len(text(free[:at-1])) ||
					strings.Contains(text(free[:at]), pattern) || regexp.QuoteMeta(pattern) != pattern {
					continue
				}
				// A token of the text's own, neither the repeat, the token after it nor one ending generation, which the
				// grammar repeats.
				other := slices.IndexFunc(free, func(id tokenizer.TokenID) bool {
					return id != repeated && id != free[at+1] && !runner.vocab.IsEOG(id)
				})
				if other < 0 {
					continue
				}
				forced := free[other]
				compile := func() *sampling.GBNFGrammar {
					grammar, err := runner.CompileLazyGBNF(fmt.Sprintf("root ::= <[%d]> <[%d]> <[%d]>+", repeated, repeated, forced), "", []string{pattern}, nil)
					if err != nil {
						t.Fatal(err)
					}
					return grammar
				}
				whole, _, wholeRows := generate(seed, compile(), false)
				card, _, cardRows := generate(seed, compile(), true)
				if !slices.Equal(card, whole) || len(card) <= at+1 || !slices.Equal(card[:at+1], free[:at+1]) || card[at+1] != forced ||
					wholeRows != len(whole) || cardRows != len(card)-at-1 {
					_ = runner.Close()
					t.Fatalf("%s seed %d trigger at %d: the top K drew %v (%d from whole rows), the whole rows %v, the free text %v",
						fixture.name, seed, at, card, cardRows, whole, free)
				}
				triggered++
				break
			}
		}
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
		if triggered == 0 {
			t.Fatalf("%s: no seed's text repeats a token a draft could carry the trigger on", fixture.name)
		}
		t.Logf("%s: %d seeds of %d triggered on a repeat", fixture.name, triggered, seeds)
	}
}
