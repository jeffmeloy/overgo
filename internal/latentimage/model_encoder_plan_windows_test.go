//go:build windows

package latentimage

import (
	"testing"

	"overgo/internal/hfgguf"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// compileModelEncoder builds the encoder's layers from first to end for a prompt of tokens from a plan of its own, as a
// test reads one range alone; the served encoder builds its ranges from the plan it keeps (modelConditioner.program).
func compileModelEncoder(catalog hfgguf.LanguageCatalog, tokens, first, end int) (*modelEncoderProgram, error) {
	plan, err := planModelEncoder(catalog)
	if err != nil {
		return nil, err
	}
	return compileModelEncoderFrom(plan, catalog, tokens, first, end)
}

// TestAPromptEncodePlansTheModelOnceALoad holds the encoder to one model-wide plan a load: a guided request's two
// prompts pass block by block, each block of each prompt built for its own token count, and the encoder planned all
// its layers again for every one of them, 64 plans of 32 layers through MOSS Ultra's encoder (SQA-511). Every block of
// two prompts of different lengths, and the whole encoder for one, are built here from one plan, as they are built
// on the card, without it.
func TestAPromptEncodePlansTheModelOnceALoad(t *testing.T) {
	t.Parallel()
	_, _, catalog := mossSmallEncoder(t, "model-small-v3-cuda.safetensors", "c96f1e0e709c9569840b10356bc55a47fd002585477fbe2e2692c37bb150786b")
	conditioner := &modelConditioner{catalog: catalog}
	blocks := int(catalog.Spec.BlockCount)
	for _, tokens := range []int{7, 12} {
		for block := range blocks {
			testcheck.Must(conditioner.program(tokens, block, block+1))(t)
		}
	}
	testcheck.Must(conditioner.program(7, 0, blocks))(t)
	if conditioner.plans != 1 {
		t.Fatalf("building %d blocks of two prompts and the whole encoder once planned the model %d times, want once", 2*blocks, conditioner.plans)
	}
}

// TestARecurrentLayerStartsFromZerosWithNoPayload holds the encoder's recurrent layers to zero states the card makes:
// each started a prompt from float32 zeros staged and uploaded, 100.5 MiB a guided request's two prompts through MOSS
// Ultra's encoder (SQA-515). Every zero feed of every block is a shape-only zero, with no host payload.
func TestARecurrentLayerStartsFromZerosWithNoPayload(t *testing.T) {
	t.Parallel()
	_, _, catalog := mossSmallEncoder(t, "model-small-v3-cuda.safetensors", "c96f1e0e709c9569840b10356bc55a47fd002585477fbe2e2692c37bb150786b")
	plan := testcheck.Must(planModelEncoder(catalog))(t)
	zeros := 0
	for block := range int(catalog.Spec.BlockCount) {
		p := testcheck.Must(compileModelEncoderFrom(plan, catalog, 9, block, block+1))(t)
		for node, value := range p.feeds {
			if node == p.input {
				continue
			}
			zeros++
			if value.Storage != reference.ValueImplicitZero || len(value.Data) != 0 {
				t.Fatalf("block %d's feed %s holds %d host values, want a shape-only zero the card makes", block, node.Name, len(value.Data))
			}
		}
	}
	if zeros == 0 {
		t.Fatal("the fixture's encoder starts no recurrent layer from zeros, so nothing was held")
	}
}
