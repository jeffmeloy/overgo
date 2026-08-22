package inference

import (
	"slices"
	"testing"

	"overgo/internal/projector"
	"overgo/internal/tensor"
	"overgo/internal/tokenizer"
)

func TestCompileProjectedInputsRejectsWidthMismatch(t *testing.T) {
	if _, _, err := CompileProjectedInputs(
		projector.MultimodalPrompt{EmbeddingWidth: tensor.SingletonExtent}, tensor.PairedExtent,
	); err == nil {
		t.Fatal("projector/model width mismatch accepted")
	}
}

func TestCompileProjectedInputsExpandsDeepstack(t *testing.T) {
	tokens := []tokenizer.TokenID{1, 2, 3, 4, 5}
	indices := []uint32{1, 3}
	width := int(tensor.PairedExtent)
	stream := make([]float32, len(indices)*width)
	for index := range indices {
		stream[index*width] = float32(index + tensor.SingletonExtent)
	}
	_, inputs, err := CompileProjectedInputs(projector.MultimodalPrompt{
		TokenIDs: tokens, Embeddings: make([]float32, len(stream)),
		DeepstackEmbeddings: [][]float32{stream}, EmbeddingWidth: width,
		EmbeddingTokenIndices: indices,
	}, uint32(width))
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs.DeepstackEmbeddings) != tensor.SingletonExtent {
		t.Fatalf("deepstack stream count = %d", len(inputs.DeepstackEmbeddings))
	}
	deepstack := inputs.DeepstackEmbeddings[tensor.FirstOffset]
	gotWidth, gotRows, valid := deepstack.MatrixExtents()
	if !valid || gotWidth != width || gotRows != len(tokens) {
		t.Fatalf("deepstack shape = %v", deepstack.Shape.Slice())
	}
	want := make([]float32, len(tokens)*width)
	for index, tokenIndex := range indices {
		copy(want[int(tokenIndex)*width:], stream[index*width:(index+tensor.SingletonExtent)*width])
	}
	if !slices.Equal(deepstack.Data, want) {
		t.Fatalf("deepstack data = %v, want %v", deepstack.Data, want)
	}
}
