package hfconvert

import (
	"cmp"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/testskip"
)

// TestDeepSeekOCRProjectorReadsBackThroughItsRuntime holds the mmproj the
// driver is about to write to being readable by the projector runtime that
// will consume it: the reader derives both towers, the window and the tiling
// from the metadata, and finds every tensor it names with the shape it
// requires. The adapter that builds it never reaches the runtime, which
// carries the device; this package does.
func TestDeepSeekOCRProjectorReadsBackThroughItsRuntime(t *testing.T) {
	t.Parallel()
	directory := cmp.Or(os.Getenv("OVERGO_DEEPSEEK_OCR_CHECKPOINT"), filepath.Join("C:/Users/jeffm/adaptive_new/models", "Unlimited-OCR"))
	if _, err := os.Stat(filepath.Join(directory, "config.json")); err != nil {
		t.Skip(testskip.Inapplicable + ": no DeepSeek-OCR checkpoint at " + directory)
	}
	repository, err := hfrepo.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	metadata, tensors, err := hfgguf.DeepSeekOCRProjectorConversion(repository)
	if err != nil {
		t.Fatal(err)
	}
	if err := requireDeepSeekOCRProjector(metadata, tensors); err != nil {
		t.Fatal(err)
	}
	if err := requireDeepSeekOCRProjector(metadata, tensors[:len(tensors)-1]); err == nil {
		t.Fatal("an mmproj missing a tensor read back")
	}
}
