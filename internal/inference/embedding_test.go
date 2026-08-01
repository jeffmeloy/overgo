package inference

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

func TestMeanPoolNormalized(t *testing.T) {
	hidden, err := reference.NewValue(
		tensor.MustShape(3, 2),
		[]float32{1, 2, 3, 3, 2, 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	embedding, err := meanPoolNormalized(hidden)
	if err != nil {
		t.Fatal(err)
	}
	want := float32(1 / math.Sqrt(3))
	for index, value := range embedding {
		if difference := math.Abs(float64(value - want)); difference > 1e-6 {
			t.Fatalf("embedding[%d] = %v, want %v", index, value, want)
		}
	}
}

func TestEmbedTokensRejectsEmptyAndOutOfRangeInput(t *testing.T) {
	runner := &Runner{vocab: &tokenizer.Vocab{
		Tokens: []tokenizer.Token{{Text: "zero"}},
	}}
	if _, _, err := runner.EmbedTokens(nil, []tokenizer.TokenID{}); err == nil ||
		!strings.Contains(err.Error(), "empty") {
		t.Fatalf("empty error = %v", err)
	}
	if _, _, err := runner.EmbedTokens(nil, []tokenizer.TokenID{1}); err == nil ||
		!strings.Contains(err.Error(), "out-of-range") {
		t.Fatalf("out-of-range error = %v", err)
	}
}

func TestMeanPoolNormalizedPreservesUpstreamZero(t *testing.T) {
	hidden, _ := reference.NewValue(tensor.MustShape(2, 1), []float32{0, 0})
	got, err := meanPoolNormalized(hidden)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != 0 || got[1] != 0 {
		t.Fatalf("zero embedding = %v", got)
	}
}

func TestEmbeddingPoolingAndNormalizationModes(t *testing.T) {
	hidden, _ := reference.NewValue(
		tensor.MustShape(2, 2),
		[]float32{3, 4, -6, 8},
	)
	none, err := poolEmbeddings(hidden, EmbeddingOptions{
		Pooling:   EmbeddingPoolingNone,
		Normalize: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(none) != 2 ||
		none[0][0] != 3 ||
		none[0][1] != 4 ||
		none[1][0] != -6 ||
		none[1][1] != 8 {
		t.Fatalf("none pooling = %v", none)
	}
	last, err := poolEmbeddings(hidden, EmbeddingOptions{
		Pooling:   EmbeddingPoolingLast,
		Normalize: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(last[0][0]-(-0.6))) > 1e-6 ||
		math.Abs(float64(last[0][1]-0.8)) > 1e-6 {
		t.Fatalf("last L2 = %v", last)
	}
	raw := []float32{-2, 4}
	normalizeEmbedding(raw, -1)
	if raw[0] != -2 || raw[1] != 4 {
		t.Fatalf("no normalization = %v", raw)
	}
	maxAbsolute := []float32{-2, 4}
	normalizeEmbedding(maxAbsolute, 0)
	if math.Abs(float64(maxAbsolute[1]-32760)) > 0.01 {
		t.Fatalf("max-absolute normalization = %v", maxAbsolute)
	}
	l1 := []float32{-2, 4}
	normalizeEmbedding(l1, 1)
	if math.Abs(float64(l1[0]-(-1.0/3))) > 1e-6 ||
		math.Abs(float64(l1[1]-(2.0/3))) > 1e-6 {
		t.Fatalf("L1 normalization = %v", l1)
	}
}

func TestProjectEmbeddingVectorsHostAppliesDenseChain(t *testing.T) {
	var dense2, dense3 bytes.Buffer
	if err := binary.Write(&dense2, binary.LittleEndian, []float32{1, 0, 0, 1, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&dense3, binary.LittleEndian, []float32{1, 1, 0, 0, 0, 1}); err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := gguf.Write(
		&encoded,
		nil,
		[]gguf.TensorData{
			{Name: "dense_2.weight", Shape: []uint64{2, 3}, Type: gguf.DTypeF32, Data: bytes.NewReader(dense2.Bytes())},
			{Name: "dense_3.weight", Shape: []uint64{3, 2}, Type: gguf.DTypeF32, Data: bytes.NewReader(dense3.Bytes())},
		},
		gguf.WriteOptions{},
	); err != nil {
		t.Fatal(err)
	}
	reader := bytes.NewReader(encoded.Bytes())
	file, err := gguf.Parse(reader, uint64(encoded.Len()), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	dense2Info, ok := file.Tensor("dense_2.weight")
	if !ok {
		t.Fatal("dense-2 projection tensor is missing")
	}
	dense3Info, ok := file.Tensor("dense_3.weight")
	if !ok {
		t.Fatal("dense-3 projection tensor is missing")
	}
	runner := &Runner{file: file, weights: model.Weights{Dense2Output: &dense2Info, Dense3Output: &dense3Info}}
	got, err := runner.projectEmbeddingVectors(context.Background(), [][]float32{{2, 3}, {-1, 4}})
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{5, 5}, {3, 3}}
	for vector := range want {
		for index := range want[vector] {
			if got[vector][index] != want[vector][index] {
				t.Fatalf("projection = %v, want %v", got, want)
			}
		}
	}
}
