package hfconvert

import (
	"testing"

	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/modeltest"
	"overgo/internal/testcheck"
)

// TestDeepSeekOCRProjectorReadsBackThroughItsRuntime holds the mmproj the
// driver is about to write to being readable by the projector runtime that
// will consume it: the reader derives both towers, the window and the tiling
// from the metadata, and finds every tensor it names with the shape it
// requires. The adapter that builds it never reaches the runtime, which
// carries the device; this package does.
func TestDeepSeekOCRProjectorReadsBackThroughItsRuntime(t *testing.T) {
	t.Parallel()
	repository := testcheck.Must(hfrepo.Open(modeltest.Directory(t, "Unlimited-OCR")))(t)
	defer repository.Close()
	metadata, tensors, err := hfgguf.DeepSeekOCRProjectorConversion(repository)
	testcheck.NoError(t, err)
	testcheck.NoError(t, requireDeepSeekOCRProjector(metadata, tensors))
	if err := requireDeepSeekOCRProjector(metadata, tensors[:len(tensors)-1]); err == nil {
		t.Fatal("an mmproj missing a tensor read back")
	}
}
