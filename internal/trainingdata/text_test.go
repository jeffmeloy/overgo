package trainingdata

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/tokenizer"
)

func TestAdjacentTokenRows(t *testing.T) {
	input, target, err := AdjacentTokenRows([]tokenizer.TokenID{3, 5, 8})
	if err != nil || !slices.Equal(input, []uint32{3, 5}) || !slices.Equal(target, []uint32{5, 8}) {
		t.Fatalf("input=%v target=%v err=%v", input, target, err)
	}
	if _, _, err := AdjacentTokenRows([]tokenizer.TokenID{3}); err == nil {
		t.Fatal("short token sequence passed")
	}
}

func TestJSONTextPairProcessor(t *testing.T) {
	processor, err := JSONTextPairProcessor("question", "answer")
	if err != nil {
		t.Fatal(err)
	}
	example, err := processor(context.Background(), RawRecord{
		ID: "gsm8k/0", Data: []byte(`{"question":"What is 48 + 24?","answer":"72"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	input, target, err := TextPair(example)
	if err != nil {
		t.Fatal(err)
	}
	if input != "What is 48 + 24?" || target != "72" {
		t.Fatalf("pair = %q -> %q", input, target)
	}
}
