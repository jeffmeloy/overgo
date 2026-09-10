//overgo:runtime-inputs caller

package tokenizer

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

type PaddedBatch struct {
	Tokens  [][]TokenID
	Lengths []uint32
}

type PaddedBatchLayout struct {
	Width   uint32
	Lengths []uint32
}

func (b PaddedBatch) Layout(wantRows int) (PaddedBatchLayout, error) {
	if len(b.Tokens) == 0 || len(b.Lengths) != len(b.Tokens) {
		return PaddedBatchLayout{}, errors.New("padded token batch layout is invalid")
	}
	if wantRows > 0 && len(b.Tokens) != wantRows {
		return PaddedBatchLayout{}, fmt.Errorf("padded token batch has %d rows, need %d", len(b.Tokens), wantRows)
	}
	width := len(b.Tokens[0])
	if width == 0 || width > math.MaxUint32 {
		return PaddedBatchLayout{}, errors.New("padded token batch width is invalid")
	}
	for index, row := range b.Tokens {
		if len(row) != width || !validPaddedLength(b.Lengths[index], uint32(width)) {
			return PaddedBatchLayout{}, fmt.Errorf(
				"padded token batch row %d layout is invalid", index,
			)
		}
	}
	return PaddedBatchLayout{Width: uint32(width), Lengths: slices.Clone(b.Lengths)}, nil
}

func (l PaddedBatchLayout) Valid(rows int) bool {
	return l.Width != 0 && len(l.Lengths) == rows &&
		!slices.ContainsFunc(l.Lengths, func(length uint32) bool {
			return !validPaddedLength(length, l.Width)
		})
}

func validPaddedLength(length, width uint32) bool { return length != 0 && length <= width }

func (l PaddedBatchLayout) Clone() PaddedBatchLayout {
	l.Lengths = slices.Clone(l.Lengths)
	return l
}
