//go:build windows && (models || modeltest)

package inference

import (
	"fmt"
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestFlashNextWindowRowsAreTheSingleTokenSteps holds the round's verify windows (strata-port/decode) to Strata's rule
// that each row of a window is bitwise the single-token step: from one prompt read, a chat decoded plainly, one-token
// windows, leaves each position a final residual row, and the same chat decoded through windows of two, three and four,
// each draft the plain decoding's own next token so every window keeps every row, leaves each position the same row,
// bit for bit. Plain decoding twice gives the same rows. Each request reads its prompt afresh: a read that reuses an
// earlier request's cache, or keeps its rows' streams for the drafter, sums otherwise (TestFlashNextDecodeRoundIsStratas).
func TestFlashNextWindowRowsAreTheSingleTokenSteps(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	model, draft := roots.ResolveModelPath(strataRoundModel), roots.ResolveModelPath(strataDraftModel)
	runner, err := openFixtureRunnerWithResidency(model, OpenOptions{HoldExperts: true, Draft: draft, StrataRound: true}, recipe.ResidencyHybridNative)
	if err != nil {
		t.Skipf("the windows' check needs %s and %s: %v", model, draft, err)
	}
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	none := 0.0
	runner.strataTestShare = &none
	runner.qsaStreamedContext = strataRoundBenchContext
	prompt, err := runner.FormatChatWithOptions([]ChatMessage{{Role: ChatRoleUser, Content: "Explain how a refrigerator keeps food cold."}},
		ChatFormatOptions{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	const generated = 48
	width := strataRoundStreams * strataRoundEmbedding
	// decode reads the prompt afresh and decodes it through windows the drafts give, nil for plain decoding, returning
	// the tokens and each position's row.
	decode := func(drafts strataDraftSource) ([]tokenizer.TokenID, map[int64][]float32) {
		t.Helper()
		rows := map[int64][]float32{}
		runner.strataTestRows = func(position int64, values []float32) {
			for row := range len(values) / width {
				rows[position+int64(row)] = values[row*width : (row+1)*width]
			}
		}
		runner.strataTestDrafts = drafts
		defer func() { runner.strataTestRows, runner.strataTestDrafts = nil, nil }()
		if _, err := runner.yieldPromptCaches(t.Context(), nil); err != nil {
			t.Fatal(err)
		}
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		ids, _, err := runner.Generate(t.Context(), prompt, GenerateOptions{
			MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, ContinueAfterEOG: true, PlainDecode: drafts == nil, StrataRound: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids, rows
	}
	plain, plainRows := decode(nil)
	if len(plainRows) != generated {
		t.Fatalf("plain decoding left %d rows for %d tokens", len(plainRows), generated)
	}
	same := func(name string, rows map[int64][]float32) {
		t.Helper()
		for position, want := range plainRows {
			got, ok := rows[position]
			if !ok {
				t.Fatalf("%s left no row at position %d", name, position)
			}
			for index := range want {
				if math.Float32bits(got[index]) != math.Float32bits(want[index]) {
					t.Fatalf("%s: position %d's row parts from plain decoding's at value %d, %g against %g", name, position, index, got[index], want[index])
				}
			}
		}
	}
	_, again := decode(nil)
	same("plain decoding again", again)
	for _, size := range []int{2, 3, 4} {
		known := func(ids []tokenizer.TokenID, want int) []tokenizer.TokenID {
			return slices.Clone(plain[len(ids):min(len(plain), len(ids)+min(want, size-1))])
		}
		ids, rows := decode(known)
		if !slices.Equal(ids, plain) {
			t.Fatalf("windows of %d give %v where plain decoding gives %v", size, ids, plain)
		}
		same(fmt.Sprintf("windows of %d", size), rows)
	}
}
