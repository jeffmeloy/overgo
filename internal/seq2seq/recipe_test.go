package seq2seq

import (
	"testing"

	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/textgeneration"
	"overgo/internal/tokenizer"
)

func generateThroughRecipe(t testing.TB, generator *Generator, request textgeneration.Request) (string, error) {
	t.Helper()
	fixture := modelrecipetest.NewCapability(t, "seq2seq-model", recipe.TaskSeq2Seq)
	if err := RegisterRuntime(fixture.Runtime, fixture.Model, generator); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.ExecuteScalar("seq2seq/runtime", request)
	if err != nil {
		return "", err
	}
	return modelrecipetest.Output[string](t, result, fixture.Program.Definition().Outputs[0].Name), nil
}

// TestGenerateAnswersWhatTheRecipeStagesAnswer holds the generator's direct
// generation to the text the recipe's three stages select for the same request.
func TestGenerateAnswersWhatTheRecipeStagesAnswer(t *testing.T) {
	const text, bound = "cab", 6
	model := tinyModel()
	pieces := make([]tokenizer.UnigramPiece, model.Dims.Vocab)
	for index := range pieces {
		pieces[index] = tokenizer.UnigramPiece{Piece: string(rune('a' + index))}
	}
	pieces[0].Piece = "▁"
	table, err := tokenizer.NewUnigram(pieces, 0)
	if err != nil {
		t.Fatal(err)
	}
	model.Dims.EOSToken = model.Dims.Vocab - 1
	generator := &Generator{model: model, tokenizer: table}
	request := textgeneration.Request{Text: text, MaxTokens: bound}
	direct, err := generator.Generate(request)
	if err != nil {
		t.Fatal(err)
	}
	staged, err := generateThroughRecipe(t, generator, request)
	if err != nil {
		t.Fatal(err)
	}
	if direct != staged || direct == "" {
		t.Fatalf("direct generation %q, recipe stages %q", direct, staged)
	}
	if _, err := generator.Generate(textgeneration.Request{Text: text}); err == nil {
		t.Fatal("a request with no token bound was answered")
	}
}

// TestGeneratorReportsTheTokensItsSelectionDecoded runs a tiny generator
// through the recipe's three stages and holds the count it reports to the
// tokens its select stage decoded, the count a measured attempt records
// against that stage's wall. The tiny model's greedy token is always its
// start token, so a selection runs to the request's bound unless that token
// ends it: the generator reports the bound's worth of tokens, then none
// under the same bound, each the count of its last selection.
func TestGeneratorReportsTheTokensItsSelectionDecoded(t *testing.T) {
	const text, bound = "cab", 6
	model := tinyModel()
	pieces := make([]tokenizer.UnigramPiece, model.Dims.Vocab)
	for index := range pieces {
		pieces[index] = tokenizer.UnigramPiece{Piece: string(rune('a' + index))}
	}
	pieces[0].Piece = "▁"
	table, err := tokenizer.NewUnigram(pieces, 0)
	if err != nil {
		t.Fatal(err)
	}
	source, err := table.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	generator := &Generator{model: model, tokenizer: table}
	for _, test := range []struct{ end, tokens int }{{end: model.Dims.Vocab - 1, tokens: bound}, {end: model.Dims.StartToken}} {
		model.Dims.EOSToken = test.end
		selected, err := executeGenerationStages(model, source, bound)
		if err != nil {
			t.Fatal(err)
		}
		output, err := generateThroughRecipe(t, generator, textgeneration.Request{Text: text, MaxTokens: bound})
		if err != nil {
			t.Fatal(err)
		}
		if len(selected) != test.tokens || output != table.Decode(selected) || generator.OutputTokens() != uint64(test.tokens) {
			t.Fatalf("ending on token %d the generator reports %d tokens for %q; want the %d its selection decoded (%v, %q)",
				test.end, generator.OutputTokens(), output, test.tokens, selected, table.Decode(selected))
		}
	}
}
