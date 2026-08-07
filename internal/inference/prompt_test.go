package inference

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/tokenizer"
)

func TestPromptTokenIDsPreservesExactSequence(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{vocab: &tokenizer.Vocab{
		Tokens: []tokenizer.Token{
			{Text: "zero"},
			{Text: "<control>", Type: tokenizer.TokenControl},
			{Text: "two"},
		},
	}}}
	input := []tokenizer.TokenID{2, 1, 0}
	ids, err := runner.promptTokenIDs("ignored", GenerateOptions{PromptTokenIDs: input})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, input) {
		t.Fatalf("ids = %v", ids)
	}
	input[0] = 0
	if ids[0] != 2 {
		t.Fatal("exact prompt token IDs alias caller storage")
	}
}

func TestPromptTokenIDsRejectsEmptyAndOutOfRangeSequences(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{vocab: &tokenizer.Vocab{
		Tokens: []tokenizer.Token{{Text: "zero"}},
	}}}
	if _, err := runner.promptTokenIDs(
		"",
		GenerateOptions{PromptTokenIDs: []tokenizer.TokenID{}},
	); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("empty sequence error = %v", err)
	}
	if _, err := runner.promptTokenIDs(
		"",
		GenerateOptions{PromptTokenIDs: []tokenizer.TokenID{1}},
	); err == nil || !strings.Contains(err.Error(), "out-of-range") {
		t.Fatalf("out-of-range error = %v", err)
	}
}
