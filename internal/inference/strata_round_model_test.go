//go:build windows && (models || modeltest)

package inference

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// strataRoundModel is Flash-Next's first shard under a model root, and strataRoundOpening the opening tokens the round's
// decoding and Overgo's agree on.
var strataRoundModel = filepath.Join("Qwen3.8-Flash-Next", "unsloth-UD-IQ4_XS", "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")

const (
	strataRoundOpening      = 8
	strataRoundBenchContext = 131072
)

// TestFlashNextDecodeRoundIsStratas holds Strata's decode round, ported as one unit (strata-port/decode), on Flash-Next:
// its plain decoding - one-token windows through the verifier, the host pool serving each layer's experts - gives the
// greedy tokens Overgo's own decoding gives for a chat; its windows of the pending token and up to three drafts, some
// drafts wrong, give its plain decoding's tokens, each window committing the tokens it kept.
func TestFlashNextDecodeRoundIsStratas(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	model := roots.ResolveModelPath(strataRoundModel)
	if _, err := os.Stat(model); err != nil {
		t.Skipf("the round's check needs %s: %v", model, err)
	}
	draftFile := roots.ResolveModelPath(strataDraftModel)
	if _, err := os.Stat(draftFile); err != nil {
		t.Skipf("the round's check needs %s: %v", draftFile, err)
	}
	// The experts stay where the load placed them and the PCIe share is none, so plain decoding and the windows compute
	// each expert in the same place (a host expert on the CPU, a card expert on the card): Strata's PCIe share takes a
	// share of each window's misses, which a window's size moves.
	runner, err := openFixtureRunnerWithResidency(model, OpenOptions{HoldExperts: true, Draft: draftFile, StrataRound: true},
		recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	none := 0.0
	runner.strataTestShare = &none
	// The bench's session: --max-context 131072 (a longer one selects with the wide top-k kernel, as Strata's would).
	runner.qsaStreamedContext = strataRoundBenchContext
	prompt, err := runner.FormatChatWithOptions(
		[]ChatMessage{{Role: ChatRoleUser, Content: "Explain how a refrigerator keeps food cold."}},
		ChatFormatOptions{AddGenerationPrompt: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	const generated = 48
	decode := func(round, plain bool, drafts strataDraftSource, spans func(int, int)) []tokenizer.TokenID {
		t.Helper()
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		runner.strataTestDrafts = drafts
		var delivered []tokenizer.TokenID
		ids, _, err := runner.Generate(t.Context(), prompt, GenerateOptions{
			MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, ContinueAfterEOG: true, PlainDecode: plain,
			StrataRound: round, OnDraftSpan: spans,
			OnToken: func(event TokenEvent) error { delivered = append(delivered, event.ID); return nil },
		})
		runner.strataTestDrafts = nil
		if err != nil {
			t.Fatal(err)
		}
		// the tokens reach the callback in order, every one before the request ends
		if !slices.Equal(delivered, ids[len(ids)-generated:]) {
			t.Fatalf("the callback saw %v of the request's %v", delivered, ids[len(ids)-generated:])
		}
		return ids
	}
	runner.strataTestOvergo = true
	ours := decode(false, true, nil, nil)
	runner.strataTestOvergo = false
	plain := decode(true, true, nil, nil)
	// Overgo's decoding and the round's take different arithmetic (the round's PLE value and conv in Strata's bf16 and
	// f16, the experts computed where Strata's pool puts them), so a near tie may flip between them: they agree on the
	// opening tokens, where a gross error would part them at once.
	common := 0
	for common < min(len(ours), len(plain)) && ours[common] == plain[common] {
		common++
	}
	t.Logf("Overgo's decoding and the round's agree for %d of %d tokens", common, len(ours))
	if prompted := len(ours) - generated; common < prompted+strataRoundOpening {
		t.Fatalf("the round's plain decoding parts from Overgo's at token %d: %v against %v", common, plain[common:], ours[common:])
	}
	// Known drafts: plain decoding's own continuation, every third window's second draft wrong.
	windows, wrong := 0, 0
	known := func(ids []tokenizer.TokenID, want int) []tokenizer.TokenID {
		windows++
		next := plain[len(ids):min(len(plain), len(ids)+want)]
		drafts := slices.Clone(next)
		if windows%3 == 0 && len(drafts) > 1 {
			drafts[1]++
			wrong++
		}
		return drafts
	}
	speculated := decode(true, false, known, nil)
	if !slices.Equal(speculated, plain) {
		t.Fatalf("the round's windows give %v where its plain decoding gives %v", speculated, plain)
	}
	if wrong == 0 {
		t.Fatal("no window took a wrong draft")
	}
	t.Logf("%d windows over %d tokens, %d with a wrong draft", windows, generated, wrong)
	// A callback's failure ends the request with it, no token delivered after it.
	gone, delivered := errors.New("the client left"), 0
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = runner.Generate(t.Context(), prompt, GenerateOptions{
		MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, ContinueAfterEOG: true, StrataRound: true,
		OnToken: func(TokenEvent) error {
			if delivered++; delivered == strataRoundOpening {
				return gone
			}
			return nil
		},
	})
	if !errors.Is(err, gone) || delivered != strataRoundOpening {
		t.Fatalf("a callback failing at token %d delivered %d and ended with %v", strataRoundOpening, delivered, err)
	}
	// The draft layer's windows: Strata's drafter drafting each window's chain, each window sized by its drafts' min_p,
	// agree with the round's plain decoding on the opening tokens. A request that drafts reads its prompt keeping its
	// rows' streams for the drafter, a read whose final residual stands up to 0.003 from the plain read's, so a near tie
	// past the opening may fall either way ('cold.' against 'coldness.' at the 25th token in the run of 2026-10-29); the
	// windows themselves are the single-token steps bit for bit (the known drafts above,
	// TestFlashNextWindowRowsAreTheSingleTokenSteps).
	spans, offered, accepted := 0, 0, 0
	drafted := decode(true, false, nil, func(drafts, agreed int) { spans, offered, accepted = spans+1, offered+drafts, accepted+agreed })
	together := 0
	for together < min(len(drafted), len(plain)) && drafted[together] == plain[together] {
		together++
	}
	if prompted := len(plain) - generated; together < prompted+strataRoundOpening {
		t.Fatalf("the round's drafted windows part from its plain decoding at token %d: %v against %v", together, drafted[together:], plain[together:])
	}
	t.Logf("the drafter: %d windows with drafts, %d drafts offered, %d accepted", spans, offered, accepted)
	if accepted == 0 {
		t.Fatal("no draft of the drafter's was accepted")
	}
	// Sampled windows: each row drawn after the window at its own position (Philox(seed, position)), so a seed's text
	// is the same whatever the drafts were: the drafter's windows give the round's plain sampled decoding's tokens.
	sampled := func(plain bool) []tokenizer.TokenID {
		t.Helper()
		sampler, err := sampling.New(sampling.Config{Temperature: 0.8, TopK: 20, TopP: 0.95, Seed: 7})
		if err != nil {
			t.Fatal(err)
		}
		ids, _, err := runner.Generate(t.Context(), prompt, GenerateOptions{
			MaxNewTokens: generated, Sampler: sampler, ContinueAfterEOG: true, PlainDecode: plain, StrataRound: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	windowsBefore := runner.strata.windows
	if plainSampled, draftedSampled := sampled(true), sampled(false); !slices.Equal(draftedSampled, plainSampled) {
		t.Fatalf("the round's drafted sampled windows give %v where its plain sampled decoding gives %v", draftedSampled, plainSampled)
	}
	if runner.strata.windows-windowsBefore < generated {
		t.Fatalf("the sampled requests decoded %d windows through the round, not its %d tokens' worth", runner.strata.windows-windowsBefore, generated)
	}
	// A follow-up turn read through the prompt cache keeps the first turn's cells where they sit, the round's session
	// cells, as Strata's session keeps a request's common prefix: it agrees on its opening tokens with the same prompt
	// read afresh. (The fresh read makes the first turn's cells by the prompt's arithmetic, not the windows', and its own
	// cells move experts off the card to make room, so the two may part later at a near tie.)
	turn := func(prompt string, cache, plain bool) ([]tokenizer.TokenID, string, int) {
		t.Helper()
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		cached := 0
		ids, text, err := runner.Generate(t.Context(), prompt, GenerateOptions{
			MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, PlainDecode: plain, StrataRound: true,
			CachePrompt: cache, OnPromptEvaluated: func(read PromptEvaluation) { cached = read.Cached },
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids, text, cached
	}
	_, answered, _ := turn(prompt, true, false)
	followUp := strings.TrimSuffix(answered, "<|im_end|>") + "<|im_end|>\n<|im_start|>user\nAnd how does a freezer differ?<|im_end|>\n<|im_start|>assistant\n"
	kept, _, cached := turn(followUp, true, false)
	if cached == 0 {
		t.Fatal("the follow-up turn read its whole prompt, none of it from the first turn's cells")
	}
	afresh, _, _ := turn(followUp, false, true)
	agreed := 0
	for agreed < min(len(kept), len(afresh)) && kept[agreed] == afresh[agreed] {
		agreed++
	}
	if prompted := len(afresh) - generated; agreed < prompted+strataRoundOpening {
		t.Fatalf("the follow-up turn from the first's %d cells parts from its prompt read afresh at token %d: %v against %v",
			cached, agreed, kept[agreed:], afresh[agreed:])
	}
	t.Logf("the follow-up turn kept %d of its prompt's cells and agrees with it read afresh for %d of %d tokens", cached, agreed, len(afresh))
}
