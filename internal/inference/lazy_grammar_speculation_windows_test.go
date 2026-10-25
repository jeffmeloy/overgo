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

// TestLazyGrammarDecodesGreedilyUntilItsTrigger: a tool request's lazy grammar masks nothing until its trigger, so the
// hermetic hybrid decodes it on the device, speculating or not, to plain greedy's tokens through the token that
// triggers it, and the grammar chooses each token after: with each generated token in turn as the trigger, at its
// first place, and a grammar that repeats one other token after it, each answer is greedy's through the trigger and
// that token after.
func TestLazyGrammarDecodesGreedilyUntilItsTrigger(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openF32FixtureRunner(writeHermeticQwen35GGUF(t), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	const generated = 8
	prompt := []tokenizer.TokenID{1, 4, 5}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	oracle, _, err := runner.Generate(t.Context(), "", GenerateOptions{
		MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: prompt,
	})
	if err != nil {
		t.Fatal(err)
	}
	answer := oracle[len(prompt):]
	for at, trigger := range answer[:generated-1] {
		if slices.Index(answer, trigger) != at {
			// The grammar triggers at the token's first place.
			continue
		}
		// A token other than greedy's next one, which the grammar repeats after the trigger.
		forced := tokenizer.TokenID(1)
		if answer[at+1] == forced {
			forced = 2
		}
		for _, speculate := range []bool{true, false} {
			grammar, err := runner.CompileLazyGBNF(fmt.Sprintf("root ::= <[%d]> <[%d]>+", trigger, forced), "", nil, []tokenizer.TokenID{trigger})
			if err != nil {
				t.Fatal(err)
			}
			sampler, err := sampling.New(sampling.Config{Temperature: 0, GBNF: grammar})
			if err != nil {
				t.Fatal(err)
			}
			if !sampler.GreedyUntilTrigger() || sampler.IsRawGreedy() {
				t.Fatal("a greedy sampler with a lazy grammar does not decode greedily until its trigger")
			}
			options := GenerateOptions{
				MaxNewTokens: generated, Sampler: sampler, DeviceGreedy: true, PromptTokenIDs: prompt,
				SpeculativeDecode: speculate, PlainDecode: !speculate,
			}
			if speculate && !runner.deviceSpeculationReady(options) {
				t.Fatal("the hybrid fixture does not speculate")
			}
			ids, _, err := runner.Generate(t.Context(), "", options)
			if err != nil {
				t.Fatalf("trigger at %d, speculating %t: %v", at, speculate, err)
			}
			want := slices.Concat(oracle[:len(prompt)+at+1], slices.Repeat([]tokenizer.TokenID{forced}, generated-at-1))
			if !slices.Equal(ids, want) {
				t.Fatalf("trigger at %d, speculating %t: decoded %v, want %v", at, speculate, ids, want)
			}
		}
	}
}

// TestLazyGrammarTriggersOnAnAcceptedDraft: on the hermetic Flash-Next with a draft layer that copies the trunk's
// selection, a draft is accepted where greedy's text repeats a token; a trigger pattern of that token twice completes
// on the accepted draft, so the span keeps its state before it, and the answer is greedy's through the repeat and the
// grammar's token after.
func TestLazyGrammarTriggersOnAnAcceptedDraft(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	trunk := writeHermeticFlashNextGGUF(t)
	runner, err := openFixtureRunnerWithResidency(trunk, OpenOptions{Draft: writeHermeticFlashNextMTPGGUF(t, trunk, true)}, recipe.ResidencyDeviceF32)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	const generated = 24
	prompt := []tokenizer.TokenID{4, 9, 17}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	oracle, _, err := runner.Generate(t.Context(), "", GenerateOptions{
		MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: prompt, ContinueAfterEOG: true, PlainDecode: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	answer := oracle[len(prompt):]
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
	triggered := 0
	for at := 1; at < generated-1; at++ {
		repeated := answer[at]
		pattern := text([]tokenizer.TokenID{repeated, repeated})
		// The repeat is the pattern's first match, ending at the repeated token.
		if answer[at-1] != repeated || strings.Index(text(answer[:at+1]), pattern) != len(text(answer[:at-1])) ||
			strings.Contains(text(answer[:at]), pattern) || regexp.QuoteMeta(pattern) != pattern {
			continue
		}
		// A token of the answer's own other than greedy's next one.
		forced := answer[slices.IndexFunc(answer, func(id tokenizer.TokenID) bool { return id != answer[at+1] })]
		grammar, err := runner.CompileLazyGBNF(fmt.Sprintf("root ::= <[%d]> <[%d]> <[%d]>+", repeated, repeated, forced), "", []string{pattern}, nil)
		if err != nil {
			t.Fatal(err)
		}
		// The sampler on the host, which reads every token's logits, is the reference.
		decode := func(device bool, onDraft func(int, int)) []tokenizer.TokenID {
			sampler, err := sampling.New(sampling.Config{Temperature: 0, GBNF: grammar})
			if err != nil {
				t.Fatal(err)
			}
			ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
				MaxNewTokens: generated, Sampler: sampler, DeviceGreedy: device, PlainDecode: !device, PromptTokenIDs: prompt,
				OnToken: func(TokenEvent) error { return nil }, OnDraftSpan: onDraft,
			})
			if err != nil {
				t.Fatalf("trigger at %d, on the device %t: %v", at, device, err)
			}
			return ids
		}
		reference := decode(false, nil)
		if !slices.Equal(reference[:len(prompt)+at+1], oracle[:len(prompt)+at+1]) || reference[len(prompt)+at+1] != forced {
			t.Fatalf("trigger at %d: the host decoded %v, greedy %v", at, reference, oracle)
		}
		accepted := 0
		if ids := decode(true, func(_, kept int) { accepted += kept }); !slices.Equal(ids, reference) || accepted == 0 {
			t.Fatalf("trigger at %d: speculated to %v with %d drafts accepted, the host to %v", at, ids, accepted, reference)
		}
		triggered++
	}
	if triggered == 0 {
		t.Fatalf("greedy's answer %v repeats no token a draft could carry the trigger on", answer)
	}
}
