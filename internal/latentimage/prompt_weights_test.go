package latentimage

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// wordTokens is a tokenizer whose tokens are a text's words, each holding the spaces before it, as a byte-level BPE's
// do; a piece it has seen keeps its id.
type wordTokens struct {
	pieces []string
}

func (w *wordTokens) Encode(text string) ([]int, error) {
	var ids []int
	for rest := text; rest != ""; {
		word := len(rest) - len(strings.TrimLeft(rest, " "))
		end := word + strings.IndexByte(rest[word:]+" ", ' ')
		piece := rest[:end]
		id := slices.Index(w.pieces, piece)
		if id < 0 {
			id = len(w.pieces)
			w.pieces = append(w.pieces, piece)
		}
		ids = append(ids, id)
		rest = rest[end:]
	}
	return ids, nil
}

func (w *wordTokens) TokenBytes(ids []int) ([][]byte, error) {
	pieces := make([][]byte, len(ids))
	for index, id := range ids {
		pieces[index] = []byte(w.pieces[id])
	}
	return pieces, nil
}

// TestAPromptWeightScalesItsTokensAttentionValue holds a prompt's weights (krea-pack/prompt-weighting) to their one
// sentence: a weighted token's attention value is scaled by its weight and no other value changes. A prompt that writes
// (text:weight) weighs its stretches, its clean text tokenized whole and each token's row carrying its stretch's weight, the prompt's rows
// starting after the template's prefix; a parenthesis that weights nothing is text, a prompt that weights nothing is
// tokenized whole as before, and a weight that is no positive number is refused. In the denoiser, weights of one give
// the unweighted program's velocity bit for bit, a weight on a prompt token moves it, and a weight on a padding row,
// whose key no query attends, moves nothing.
func TestAPromptWeightScalesItsTokensAttentionValue(t *testing.T) {
	t.Parallel()
	spans := testcheck.Must(weightedSpans("a (red:2) fox (in snow:0.5)"))(t)
	want := []weightedSpan{{"a ", unweighted}, {"red", 2}, {" fox ", unweighted}, {"in snow", 0.5}}
	if !slices.Equal(spans, want) {
		t.Fatalf("the prompt split into %v; want %v", spans, want)
	}
	for _, plain := range []string{"a red fox", "a fox (by the river)", "a box (0, 0, 16, 16)"} {
		if spans := testcheck.Must(weightedSpans(plain))(t); spans != nil {
			t.Errorf("%q, which weights nothing, split into %v", plain, spans)
		}
	}
	if _, err := weightedSpans("a (red:-1) fox"); err == nil {
		t.Error("a negative weight was taken")
	}
	rows := rowWeights(10, 3, 5, testcheck.Must(alignedTokens(cleanText(spans), 0, spans, &wordTokens{}))(t).weights)
	if !slices.Equal(rows, []float32{1, 1, 1, 1, 2, 1, 0.5, 0.5, 1, 1}) {
		t.Fatalf("the rows weigh %v; want the prompt's tokens weighed after the 3 prefix rows", rows)
	}

	const gh, gw, textSeq = 2, 2, 3
	spec := syntheticSpec()
	denoiser := testcheck.Must(NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), syntheticStore(spec)))(t)
	latent, encoded := syntheticForwardInputs(spec, textSeq, gh*gw)
	patches := mustPack(t, latent, spec.InChannels, gh, gw)
	mask := []bool{true, true, false}
	velocity := func(weights []float32) []float32 {
		t.Helper()
		program := testcheck.Must(CompileDenoiserProgram(spec, 1e-5, mask, weights, gh, gw, dtype.F32, false))(t)
		return testcheck.Must(program.Forward(reference.Execute, denoiser, patches, encoded, 0.9))(t).Velocity
	}
	plain := velocity(nil)
	if ones := velocity([]float32{1, 1, 1}); !slices.Equal(ones, plain) {
		t.Error("weights of one moved the velocity")
	}
	if weighted := velocity([]float32{2, 1, 1}); slices.Equal(weighted, plain) {
		t.Error("a weight on a prompt token left the velocity where it was")
	}
	if padded := velocity([]float32{1, 1, 3}); !slices.Equal(padded, plain) {
		t.Error("a weight on a padding row, whose key no query attends, moved the velocity")
	}
}
