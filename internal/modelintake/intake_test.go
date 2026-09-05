package modelintake

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

// recordingGenerator answers every prompt with the prompt's words reversed,
// reporting the prompt token count the way a runner does.
type recordingGenerator struct{}

func (recordingGenerator) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	words := strings.Fields(prompt)
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: len(words)})
	}
	var ids []tokenizer.TokenID
	for index := range len(words) {
		ids = append(ids, tokenizer.TokenID(index))
	}
	for index := range min(options.MaxNewTokens, len(words)) {
		piece := words[len(words)-index-1] + " "
		if options.OnToken != nil {
			if err := options.OnToken(inference.TokenEvent{Piece: piece}); err != nil {
				return nil, "", err
			}
		}
		ids = append(ids, tokenizer.TokenID(len(words)+index))
	}
	return ids, "", nil
}

// TestRecordExactSuiteReplays pins the bootstrap golden: what the model
// produced becomes the expected text, with the prompt and generated token
// counts a compiled exact plan requires, so a replay of the same model
// reproduces it exactly.
func TestRecordExactSuiteReplays(t *testing.T) {
	suite, err := RecordExactSuite(t.Context(), recordingGenerator{}, "intake-test", []string{"one two three", "alpha beta"}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(suite.Cases) != 2 || suite.Cases[0].Text != "three two one " || suite.Cases[0].PromptTokens != 3 || suite.Cases[0].GeneratedTokens != 3 {
		t.Fatalf("recorded suite = %+v", suite.Cases)
	}
	plan, err := evaluation.CompileExact(suite)
	if err != nil {
		t.Fatalf("recorded suite does not compile: %v", err)
	}
	if !plan.Identity().Valid() {
		t.Fatal("recorded suite has no identity")
	}
	if _, err := RecordExactSuite(t.Context(), recordingGenerator{}, "", nil, 0); err == nil {
		t.Fatal("an empty recording request was accepted")
	}
}
