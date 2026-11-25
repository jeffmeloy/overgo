//go:build windows

package latentimage

import (
	"testing"

	"overgo/internal/hfgguf"
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
