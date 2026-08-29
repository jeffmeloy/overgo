package inference

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/tokenizer"
)

func TestEncoderDecoderBatchRejectsInvalidSourceMask(t *testing.T) {
	runner := &Runner{}
	_, _, err := runner.DecodeEncoderDecoderBatch(
		t.Context(),
		&EncoderDecoderBatchSession{Sequences: []*EncoderDecoderSession{{}}},
		tokenizer.PaddedBatch{
			Tokens: [][]tokenizer.TokenID{{1}}, Lengths: []uint32{1},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "source mask") {
		t.Fatalf("source-mask error = %v", err)
	}
}

func TestPaddedTokenBatchLayoutPreservesDerivedLengths(t *testing.T) {
	active := [][]tokenizer.TokenID{{1, 2}, {3, 4, 5}}
	var width int
	for _, tokens := range active {
		width += len(tokens)
	}
	batch := tokenizer.PaddedBatch{
		Tokens: make([][]tokenizer.TokenID, len(active)), Lengths: make([]uint32, len(active)),
	}
	for index, tokens := range active {
		batch.Tokens[index] = append(slices.Clone(tokens), make([]tokenizer.TokenID, width-len(tokens))...)
		batch.Lengths[index] = uint32(len(tokens))
	}
	layout, err := batch.Layout(len(active))
	if err != nil || layout.Width != uint32(width) || !slices.Equal(layout.Lengths, batch.Lengths) {
		t.Fatalf("layout = %+v error %v", layout, err)
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
