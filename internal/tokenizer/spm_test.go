package tokenizer

import (
	"slices"
	"testing"

	"overgo/internal/gguf"
)

func TestLlamaSPMEncodeDecode(t *testing.T) {
	tokens := []string{
		"<unk>", "<s>", "</s>", "▁", "h", "e", "l", "o",
		"▁h", "▁he", "▁hel", "▁hell", "▁hello",
	}
	scores := make([]float32, len(tokens))
	for index := 8; index < len(scores); index++ {
		scores[index] = float32(index)
	}
	types := make([]int32, len(tokens))
	for index := range types {
		types[index] = int32(TokenNormal)
	}
	types[0] = int32(TokenUnknown)
	types[1] = int32(TokenControl)
	types[2] = int32(TokenControl)
	file := &gguf.File{Metadata: []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		scalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		scalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
		scalar("tokenizer.ggml.unknown_token_id", gguf.ValueTypeUint32, uint32(0)),
		scalar("tokenizer.ggml.add_bos_token", gguf.ValueTypeBool, true),
		scalar("tokenizer.ggml.add_space_prefix", gguf.ValueTypeBool, true),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		array("tokenizer.ggml.scores", gguf.ValueTypeFloat32, scores),
		array("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
	}}
	vocab, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := vocab.Encode("hello", EncodeOptions{AddSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	if want := []TokenID{1, 12}; !slices.Equal(ids, want) {
		t.Fatalf("IDs = %v, want %v", ids, want)
	}
	text, err := vocab.Decode(ids, false)
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("decoded text = %q, want hello", text)
	}
	piece, err := vocab.DecodePiece(12, false)
	if err != nil {
		t.Fatal(err)
	}
	if piece != " hello" {
		t.Fatalf("streaming piece = %q, want %q", piece, " hello")
	}
}

func TestLlamaSPMByteFallback(t *testing.T) {
	tokens := []string{"<unk>", "<s>", "</s>", "▁", "<0xC3>", "<0xA9>"}
	types := []int32{
		int32(TokenUnknown),
		int32(TokenControl),
		int32(TokenControl),
		int32(TokenNormal),
		int32(TokenByte),
		int32(TokenByte),
	}
	file := &gguf.File{Metadata: []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		scalar("tokenizer.ggml.unknown_token_id", gguf.ValueTypeUint32, uint32(0)),
		scalar("tokenizer.ggml.add_space_prefix", gguf.ValueTypeBool, false),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		array("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, len(tokens))),
		array("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
	}}
	vocab, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := vocab.Encode("é", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if want := []TokenID{4, 5}; !slices.Equal(ids, want) {
		t.Fatalf("IDs = %v, want %v", ids, want)
	}
	text, err := vocab.Decode(ids, false)
	if err != nil {
		t.Fatal(err)
	}
	if text != "é" {
		t.Fatalf("decoded text = %q, want é", text)
	}
}
