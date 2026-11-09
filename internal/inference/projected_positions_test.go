package inference

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestAClipPromptReadInChunksKeepsItsLargestPosition holds a projected prompt the card reads in chunks to the cache
// position a whole read gives. A clip's multi-axis positions restart their rows and columns at each frame, so a chunk
// that ends inside a frame can hold no position as large as the chunk before it: on 2026-11-09 the judge's clip prompt,
// bisected under memory pressure, was refused there ("projected positions precede the cache's next position") and the
// judge-calibration landing failed. Read whole or in chunks of 8, 3 and 3 tokens, the prompt leaves its cache at the
// same next position; a prompt's first chunk whose positions lie before a cache it did not write is still refused.
func TestAClipPromptReadInChunksKeepsItsLargestPosition(t *testing.T) {
	t.Parallel()
	// Two tokens of text, then two frames of one row of six columns: temporal, row and column axes, the fourth unused.
	whole := MultiAxisPositions{
		{0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3},
		{0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
		{0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7},
		{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	inputs := ProjectedInputs{MultiAxisPositions: &whole}
	read := testcheck.Must(projectedPositionAfter(whole, 0, false))(t)
	var next uint32
	start := 0
	for _, count := range []int{8, 3, 3} {
		chunk := inputs.span(start, count)
		var err error
		if next, err = projectedPositionAfter(*chunk.MultiAxisPositions, next, chunk.continues); err != nil {
			t.Fatalf("the chunk of %d tokens from %d is refused: %v", count, start, err)
		}
		start += count
	}
	if next != read {
		t.Fatalf("read in chunks the prompt leaves its cache at position %d, read whole at %d", next, read)
	}
	if _, err := projectedPositionAfter(whole, read+1, false); err == nil {
		t.Fatal("a prompt's first chunk was read onto a cache it did not write, past its positions")
	}
}
