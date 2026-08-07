package inference

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/tokenizer"
)

func TestDecodeT5BatchRejectsInvalidSourceMask(t *testing.T) {
	runner := &Runner{}
	_, _, err := runner.DecodeT5Batch(
		context.Background(),
		&T5BatchSession{Sequences: []*T5Session{{}}},
		PaddedTokenBatch{
			Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{1},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "source mask") {
		t.Fatalf("source-mask error = %v", err)
	}
}

func TestValidatePaddedTokenBatchMasksSuffixes(t *testing.T) {
	batch := PaddedTokenBatch{
		Tokens: [][]tokenizer.TokenID{
			{1, 2, 0, 0},
			{3, 4, 5, 0},
		},
		Lengths: []uint32{2, 3},
	}
	width, err := validatePaddedTokenBatch(batch, 2)
	if err != nil || width != 4 {
		t.Fatalf("validation = width %d error %v", width, err)
	}
	for index, length := range batch.Lengths {
		active := batch.Tokens[index][:length]
		if len(active) != int(length) {
			t.Fatalf("row %d active tokens = %v", index, active)
		}
	}
}

func TestValidatePaddedTokenBatchRejectsInvalidMasks(t *testing.T) {
	tests := []PaddedTokenBatch{
		{},
		{Tokens: [][]tokenizer.TokenID{{1}}, Lengths: nil},
		{Tokens: [][]tokenizer.TokenID{{1}, {2, 3}}, Lengths: []uint32{1, 2}},
		{Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{0}},
		{Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{2}},
	}
	for index, batch := range tests {
		if _, err := validatePaddedTokenBatch(batch, 0); err == nil {
			t.Fatalf("invalid batch %d accepted", index)
		}
	}
	_, err := validatePaddedTokenBatch(PaddedTokenBatch{
		Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{1},
	}, 2)
	if err == nil || !strings.Contains(err.Error(), "need 2") {
		t.Fatalf("row-count error = %v", err)
	}
}
