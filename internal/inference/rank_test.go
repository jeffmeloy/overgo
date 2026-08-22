package inference

import (
	"bytes"
	"context"
	"encoding/binary"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tokenizer"
)

func TestRankPairPromptUsesNamedTemplate(t *testing.T) {
	prompt := renderClassifierPrompt("Query: {query}\nDocument: {document}", "needle", "haystack")
	if prompt != "Query: needle\nDocument: haystack" {
		t.Fatalf("rank prompt = %q", prompt)
	}
}

func TestRankPairPromptUsesConfiguredSeparators(t *testing.T) {
	vocab := &tokenizer.Vocab{
		BOS: 1, EOS: 2, SEP: 3, AddBOS: true, AddEOS: true, AddSEP: true,
	}
	prompt, err := assembleClassifierPairTokens(vocab, []tokenizer.TokenID{4}, []tokenizer.TokenID{5})
	if err != nil {
		t.Fatal(err)
	}
	want := []tokenizer.TokenID{1, 4, 2, 3, 5, 2}
	if len(prompt) != len(want) {
		t.Fatalf("rank tokens = %v", prompt)
	}
	for index := range want {
		if prompt[index] != want[index] {
			t.Fatalf("rank tokens = %v", prompt)
		}
	}
}

func TestRankAdmission(t *testing.T) {
	vocab := &tokenizer.Vocab{Tokens: []tokenizer.Token{{Text: "zero"}}}
	unsupported := &Runner{preparedModel: preparedModel{vocab: vocab, spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}}}
	unsupported = attachFixtureProgram(unsupported)
	if _, err := unsupported.RankTokens(nil, []tokenizer.TokenID{0}); err == nil ||
		!strings.Contains(err.Error(), "no pinned Qwen rank graph") {
		t.Fatalf("unsupported error = %v", err)
	}
	missing := &Runner{preparedModel: preparedModel{vocab: vocab, spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen3"}}}}
	missing = attachFixtureProgram(missing)
	if _, err := missing.RankTokens(nil, []tokenizer.TokenID{0}); err == nil ||
		!strings.Contains(err.Error(), "tensor is missing") {
		t.Fatalf("missing-head error = %v", err)
	}
}

func TestProjectRankScoresHost(t *testing.T) {
	var tensorData bytes.Buffer
	if err := binary.Write(&tensorData, binary.LittleEndian, []float32{1, 0, 0, 1}); err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := gguf.Write(&encoded, nil, []gguf.TensorData{{
		Name: "cls.output.weight", Shape: []uint64{2, 2},
		Type: gguf.DTypeF32, Data: bytes.NewReader(tensorData.Bytes()),
	}}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Parse(bytes.NewReader(encoded.Bytes()), uint64(encoded.Len()), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	info, ok := file.Tensor("cls.output.weight")
	if !ok {
		t.Fatal("classifier tensor is missing")
	}
	runner := &Runner{preparedModel: preparedModel{file: file, weights: model.Weights{ClassifierOutput: &info}}}
	scores, err := runner.projectClassifierScores(context.Background(), []float32{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 2 || scores[0] != 2 || scores[1] != 3 {
		t.Fatalf("scores = %v", scores)
	}
}
