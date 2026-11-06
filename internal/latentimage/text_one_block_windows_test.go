//go:build windows

package latentimage

import (
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/hfgguf"
	"overgo/internal/testcheck"
)

// TestATextEncoderHoldsOneBlockOnTheCard: the text encoder runs a block at a time, two prompts of different lengths
// through each block before the next, their rows kept on the card between blocks, and gives each prompt the hidden states
// its native reference holds, bit for bit, as the whole encoder's program does. What encoding holds of the card beyond the
// session is the whole program's less the weights of every block but the largest, at most, and it keeps none of them.
func TestATextEncoderHoldsOneBlockOnTheCard(t *testing.T) {
	cudatest.Require(t)
	s, root, catalog := mossSmallEncoder(t, "model-small-v3-cuda.safetensors", "c96f1e0e709c9569840b10356bc55a47fd002585477fbe2e2692c37bb150786b")
	session := testcheck.Must(graphruntime.NewResidentSession(0))(t)
	defer session.Close(t.Context())
	conditioner := modelConditioner{catalog: catalog, root: root}
	var texts []textInput
	var embeddings, want [][]float32
	for _, name := range []string{"t22", "t65"} {
		embedding := refinementValues(t, s, name+".embedding")
		text := textInput{IDs: make([]int, len(embedding)/int(catalog.Spec.EmbeddingLength))}
		text.Mask = make([]bool, len(text.IDs))
		for i := range text.Mask {
			text.Mask[i] = true
		}
		texts, embeddings, want = append(texts, text), append(embeddings, embedding), append(want, refinementValues(t, s, name+".final_norm"))
	}
	before := session.StaticBytes()
	held := func(encode func()) uint64 {
		t.Helper()
		testcheck.NoError(t, session.ReleaseStageMemory(t.Context()))
		base := driver.ProcessCurrentBytes()
		driver.ResetStagePeakBytes()
		encode()
		return driver.StagePeakBytes() - base
	}
	whole := held(func() {
		for index, text := range texts {
			hidden := testcheck.Must(conditioner.encode(t.Context(), session, text, embeddings[index], nil))(t)
			if _, unequal, _ := compareMOSSValues(hidden, want[index], mossStageTolerance{}); unequal != 0 {
				t.Fatalf("prompt %d: the whole encoder differs from its reference in %d values", index, unequal)
			}
		}
	})
	each := held(func() {
		hidden := testcheck.Must(conditioner.encodeEach(t.Context(), session, texts, embeddings))(t)
		for index := range texts {
			if _, unequal, _ := compareMOSSValues(hidden[index], want[index], mossStageTolerance{}); unequal != 0 {
				t.Fatalf("prompt %d: the encoder a block at a time differs from its reference in %d values", index, unequal)
			}
		}
	})
	weights, largest := encoderWeightBytes(t, catalog, 0, int(catalog.Spec.BlockCount)), uint64(0)
	for block := range int(catalog.Spec.BlockCount) {
		largest = max(largest, encoderWeightBytes(t, catalog, block, block+1))
	}
	t.Logf("the whole encoder held %d bytes, a block at a time %d; its weights are %d bytes, its largest block's %d", whole, each, weights, largest)
	if each+weights-largest > whole {
		t.Fatalf("a block at a time held %d bytes, more than the whole program's %d less every block's weights but the largest, %d", each, whole, weights-largest)
	}
	if session.StaticBytes() != before {
		t.Fatalf("encoding kept %d bytes of weights on the card", session.StaticBytes()-before)
	}
}

// encoderWeightBytes is what the encoder's blocks from first to end place on the card.
func encoderWeightBytes(t *testing.T, catalog hfgguf.LanguageCatalog, first, end int) uint64 {
	t.Helper()
	p := testcheck.Must(compileModelEncoder(catalog, 1, first, end))(t)
	var total uint64
	for _, node := range p.weights {
		bytes := testcheck.Must(node.Shape.Bytes(node.Type))(t)
		total += bytes
	}
	return total
}
