package trainingdata

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
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
	processor := testcheck.Must(JSONTextPairProcessor("question", "answer"))(t)
	example := testcheck.Must(processor(t.Context(), RawRecord{
		ID: "gsm8k/0", Data: []byte(`{"question":"What is 48 + 24?","answer":"72"}`),
	}))(t)
	input, target, err := TextPair(example)
	testcheck.NoError(t, err)
	if input != "What is 48 + 24?" || target != "72" {
		t.Fatalf("pair = %q -> %q", input, target)
	}
}
