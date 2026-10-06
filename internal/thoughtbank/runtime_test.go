package thoughtbank

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/hfbpe"
	"overgo/internal/textgeneration"
)

func TestGenerateRequestHasNoPackageLocalTokenCeiling(t *testing.T) {
	request := textgeneration.Request{Text: "continue", MaxTokens: 257}
	if err := textgeneration.Validate(request); err != nil {
		t.Fatalf("positive request was rejected by a package-local token ceiling: %v", err)
	}
}

// fixtureGenerator is a generator over the small fixture model with a
// one-character-per-token tokenizer, so a text's tokens are its characters.
func fixtureGenerator(t *testing.T) (*Generator, int) {
	t.Helper()
	weights, fixture := buildFixtureModel(1)
	vocabulary := make(map[string]int, fixture.vocab)
	for index := range fixture.vocab {
		vocabulary[string(rune('a'+index))] = index
	}
	body, err := json.Marshal(map[string]any{
		"added_tokens": []any{}, "model": map[string]any{"type": "BPE", "vocab": vocabulary, "merges": []any{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), body, 0o600); err != nil {
		t.Fatal(err)
	}
	tokenizer, err := hfbpe.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	return &Generator{weights: weights, config: ArchConfig{MemDim: fixture.memDim}, tokenizer: tokenizer}, fixture.vocab
}

// TestGeneratorReportsTheTokensItDecoded generates twice on a tiny model
// under two bounds and holds the count the generator reports to the tokens
// of its last generation, the count a measured attempt records against the
// generation stage's wall.
func TestGeneratorReportsTheTokensItDecoded(t *testing.T) {
	generator, _ := fixtureGenerator(t)
	for _, bound := range []int{5, 2} {
		generation, err := generator.Generate(textgeneration.Request{Text: "cab", MaxTokens: bound})
		if err != nil {
			t.Fatal(err)
		}
		if len(generation.Tokens) != bound || generator.OutputTokens() != uint64(bound) {
			t.Fatalf("generator reports %d tokens after generating %v under a bound of %d; want the %d it decoded",
				generator.OutputTokens(), generation.Tokens, bound, bound)
		}
	}
}
