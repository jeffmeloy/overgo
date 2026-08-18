package evaluation

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

const (
	exactPromptTokens    = 2
	exactGeneratedTokens = 2
	exactMaxTokens       = exactGeneratedTokens
)

type exactGenerator struct {
	pieces []string
	err    error
}

func (g exactGenerator) Generate(_ context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if g.err != nil {
		return nil, "", g.err
	}
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: exactPromptTokens})
	for index, piece := range g.pieces {
		if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(index), Piece: piece}); err != nil {
			return nil, "", err
		}
	}
	return make([]tokenizer.TokenID, exactPromptTokens+len(g.pieces)), "", nil
}

func TestCompileExactOwnsStableSuite(t *testing.T) {
	suite := exactFixture()
	first, err := CompileExact(suite)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileExact(suite)
	if err != nil {
		t.Fatal(err)
	}
	if first.Identity() != second.Identity() {
		t.Fatalf("identity differs: %s != %s", first.Identity(), second.Identity())
	}
	suite.Cases[0].Text = "changed"
	results, err := EvaluateExact(context.Background(), exactGenerator{pieces: []string{"o", "k"}}, first)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Text != "ok" {
		t.Fatalf("results = %+v", results)
	}
}

func TestCompileExactRejectsIncompleteSuite(t *testing.T) {
	for _, suite := range []ExactSuite{
		{},
		{Schema: "test/v1", Source: "fixture"},
		{Schema: "test/v1", Source: "fixture", Cases: []ExactCase{{Name: "case"}}},
	} {
		if _, err := CompileExact(suite); err == nil {
			t.Fatalf("accepted %+v", suite)
		}
	}
}

func TestExactReportsGenerationFailureAndMismatch(t *testing.T) {
	plan, err := CompileExact(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := EvaluateExact(context.Background(), exactGenerator{err: errors.New("generate")}, plan); err == nil {
		t.Fatal("generation failure accepted")
	}
	if _, err := EvaluateExact(context.Background(), exactGenerator{pieces: []string{"n", "o"}}, plan); err == nil {
		t.Fatal("mismatch accepted")
	}
}

func exactFixture() ExactSuite {
	return ExactSuite{
		Schema: "test/v1", Source: "fixture",
		Cases: []ExactCase{{
			Name: "case", Prompt: "prompt", MaxTokens: exactMaxTokens, Text: "ok",
			PromptTokens: exactPromptTokens, GeneratedTokens: exactGeneratedTokens,
		}},
	}
}
