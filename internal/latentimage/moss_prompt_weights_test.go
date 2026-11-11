package latentimage

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestAPromptWeightScalesMOSSTokensAttentionValue holds MOSS's prompt weights (krea-pack/moss-weighting): MOSS renders
// its prompt through its chat template, so a weighted prompt is rendered around a mark, and the text the template puts
// before it, each stretch and the text after are tokenized apart, every row but a stretch's weighing one; a prompt that
// weights nothing is rendered and tokenized whole as before, with no weights. In the denoiser on the golden's weights,
// weights of one give the unweighted output bit for bit, and a weight on a text row moves it.
func TestAPromptWeightScalesMOSSTokensAttentionValue(t *testing.T) {
	t.Parallel()
	render := func(content string) (string, error) { return "<user> " + content + " <assistant>", nil }
	words := func(text string) ([]int, error) { return make([]int, len(strings.Fields(text))), nil }
	weighted := testcheck.Must(chatTokens("a (red:2) fox", render, words))(t)
	if !slices.Equal(weighted.weights, []float32{1, 1, 2, 1, 1}) || len(weighted.ids) != len(weighted.weights) {
		t.Fatalf("a weighted prompt's rendering weighs %v over %d tokens; want the template's tokens and the plain words at one, red at two", weighted.weights, len(weighted.ids))
	}
	if plain := testcheck.Must(chatTokens("a red fox", render, words))(t); plain.weights != nil || len(plain.ids) != len(strings.Fields("<user> a red fox <assistant>")) {
		t.Fatalf("a prompt that weights nothing gave %+v; want the whole rendering's tokens and no weights", plain)
	}

	const text = 4
	output := func(weights ...float32) []float32 {
		t.Helper()
		program := testcheck.Must(compileRotaryDenoiserProgram(denoiserSpec(), 1e-6, 2, 3, text, nil, dtype.F32, weights...))(t)
		got := testcheck.Must(reference.Execute([]*tensor.Tensor{program.output}, denoiserFeeds(t, program, 0, dtype.F32)))(t)
		return got[program.output].Data
	}
	plain := output()
	if ones := output(1, 1, 1, 1); !slices.Equal(ones, plain) {
		t.Error("weights of one moved the output")
	}
	if moved := output(3, 1, 1, 1); slices.Equal(moved, plain) {
		t.Error("a weight on a text row left the output where it was")
	}
}
