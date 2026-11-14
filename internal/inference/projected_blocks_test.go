package inference

import (
	"slices"
	"testing"

	"overgo/internal/model"
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

// TestAProjectedPromptTheCardCannotTakeWholeRunsOnTheHost holds the card's admission of a projected prompt to the
// prefill's own reading of it (SQA-364): over every prompt length and chunk, the card takes a prompt with bidirectional
// blocks exactly where every chunk the prefill cuts it into carries them (span), so a prompt it would read in part and
// refuse at a later chunk is read on the host, chosen before any device work. A prompt of 9 tokens whose blocks end by
// its 8th, in chunks of 8, was admitted, read its first chunk on the card and was refused at its second.
func TestAProjectedPromptTheCardCannotTakeWholeRunsOnTheHost(t *testing.T) {
	t.Parallel()
	runner := attachFixtureProgram(&Runner{preparedModel: preparedModel{spec: model.Spec{
		CommonSpec: model.CommonSpec{Architecture: "gemma4", ContextLength: 16, EmbeddingLength: 4},
	}}})
	inputs := ProjectedInputs{BidirectionalAttentionBlocks: []AttentionBlock{{Start: 2, End: 5}, {Start: 6, End: 8}}}
	for tokens := 8; tokens <= 12; tokens++ {
		for chunk := 1; chunk <= 12; chunk++ {
			runner.promptChunkTokens = chunk
			read := true
			for start := 0; start < tokens; start += chunk {
				read = read && !inputs.span(start, min(chunk, tokens-start)).blocksSplit
			}
			if admitted := runner.deviceProjectable(inputs, tokens); admitted != read {
				t.Errorf("a prompt of %d tokens in chunks of %d: the card admits it %t, where its prefill reads it whole %t", tokens, chunk, admitted, read)
			}
		}
	}
	if runner.promptChunkTokens = 4; !runner.deviceProjectable(ProjectedInputs{}, 12) {
		t.Error("a prompt with no blocks, longer than a chunk, was sent to the host")
	}
}
