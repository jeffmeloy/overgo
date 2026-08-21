package inference

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/tokenizer"
)

func TestEncoderDecoderBatchRejectsInvalidSourceMask(t *testing.T) {
	runner := &Runner{}
	_, _, err := runner.DecodeEncoderDecoderBatch(
		context.Background(),
		&EncoderDecoderBatchSession{Sequences: []*EncoderDecoderSession{{}}},
		tokenizer.PaddedBatch{
			Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{1},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "source mask") {
		t.Fatalf("source-mask error = %v", err)
	}
}

func TestValidatePaddedTokenBatchMasksSuffixes(t *testing.T) {
	batch := tokenizer.PaddedBatch{
		Tokens: [][]tokenizer.TokenID{
			{1, 2, 0, 0},
			{3, 4, 5, 0},
		},
		Lengths: []uint32{2, 3},
	}
	layout, err := batch.Layout(2)
	if err != nil || layout.Width != 4 {
		t.Fatalf("validation = width %d error %v", layout.Width, err)
	}
	for index, length := range batch.Lengths {
		active := batch.Tokens[index][:length]
		if len(active) != int(length) {
			t.Fatalf("row %d active tokens = %v", index, active)
		}
	}
}

func TestValidatePaddedTokenBatchRejectsInvalidMasks(t *testing.T) {
	tests := []tokenizer.PaddedBatch{
		{},
		{Tokens: [][]tokenizer.TokenID{{1}}, Lengths: nil},
		{Tokens: [][]tokenizer.TokenID{{1}, {2, 3}}, Lengths: []uint32{1, 2}},
		{Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{0}},
		{Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{2}},
	}
	for index, batch := range tests {
		if _, err := batch.Layout(0); err == nil {
			t.Fatalf("invalid batch %d accepted", index)
		}
	}
	_, err := (tokenizer.PaddedBatch{
		Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{1},
	}).Layout(2)
	if err == nil || !strings.Contains(err.Error(), "need 2") {
		t.Fatalf("row-count error = %v", err)
	}
}
