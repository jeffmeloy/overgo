package inference

import (
	"slices"
	"testing"
)

// TestABidirectionalBlockIsReadWholeOrRefused holds how the card reads a prompt's bidirectional attention blocks
// (judge-speed/cell-blocks). A token of a block reads to the block's last cell and any other to its own, from the
// prompt's first cell. A chunk that holds the prompt from its start and every block whole carries the blocks; a chunk
// that cuts one, or starts later, carries none and is marked, so the card refuses to read it.
func TestABidirectionalBlockIsReadWholeOrRefused(t *testing.T) {
	t.Parallel()
	blocks := []AttentionBlock{{Start: 2, End: 5}, {Start: 6, End: 8}}
	if got, want := cellReach(10, 9, blocks), []uint32{10, 11, 14, 14, 14, 15, 17, 17, 18}; !slices.Equal(got, want) {
		t.Fatalf("the cells read to %v, want %v", got, want)
	}
	inputs := ProjectedInputs{BidirectionalAttentionBlocks: blocks}
	if whole := inputs.span(0, 9); whole.blocksSplit || !slices.Equal(whole.BidirectionalAttentionBlocks, blocks) {
		t.Fatalf("a chunk holding the prompt whole carries %+v, split %t; want every block", whole.BidirectionalAttentionBlocks, whole.blocksSplit)
	}
	for _, chunk := range [][2]int{{0, 4}, {4, 5}} {
		if cut := inputs.span(chunk[0], chunk[1]); !cut.blocksSplit || len(cut.BidirectionalAttentionBlocks) != 0 {
			t.Errorf("a chunk of %d tokens from %d carries %+v, split %t; want none, marked", chunk[1], chunk[0], cut.BidirectionalAttentionBlocks, cut.blocksSplit)
		}
	}
	if plain := (ProjectedInputs{}).span(0, 4); plain.blocksSplit {
		t.Error("a chunk of a prompt with no blocks is marked")
	}
}
