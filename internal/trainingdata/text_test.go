package trainingdata

import (
	"context"
	"testing"
)

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
