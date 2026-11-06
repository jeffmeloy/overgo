package tokenizer

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestUnigramDecodeRestoresTextAndBytes(t *testing.T) {
	table := testcheck.Must(NewUnigram([]UnigramPiece{
		{Piece: "<unk>"},
		{Piece: "\u2581hello"},
		{Piece: "\u2581world"},
		{Piece: "<0xE2>"},
		{Piece: "<0x82>"},
		{Piece: "<0xAC>"},
	}, 0))(t)
	if got := table.Decode([]int{1, 2, 3, 4, 5}); got != "hello world€" {
		t.Fatalf("decoded text = %q", got)
	}
}
