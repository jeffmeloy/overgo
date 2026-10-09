package hfgguf

import (
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/modeltest"
)

// deepSeekOCRCheckpoint locates a DeepSeek-OCR family checkpoint for the
// conversion tests, or explains why none applies here.
func deepSeekOCRCheckpoint(t *testing.T) *hfrepo.Repository {
	t.Helper()
	repository, err := hfrepo.Open(modeltest.Directory(t, "Unlimited-OCR"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	return repository
}

// TestDeepSeekOCRConversionValidatesCheckpoint holds the language conversion
// to a catalog the loader reads: the decoder's spec resolves against the
// deepseek2-ocr profile and every weight it requires is present with its
// shape, including the stacked routed experts and the shared experts.
func TestDeepSeekOCRConversionValidatesCheckpoint(t *testing.T) {
	repository := deepSeekOCRCheckpoint(t)
	if !IsDeepSeekOCRRepository(repository.Identity) {
		t.Fatalf("identity %+v is not the DeepSeek-OCR family", repository.Identity)
	}
	spec, err := ValidateDeepSeekOCRRepository(repository)
	if err != nil {
		t.Fatal(err)
	}
	if spec.BlockCount == 0 || spec.ExpertCount == 0 || spec.ExpertUsedCount == 0 || spec.SharedExpertCount == 0 || spec.LeadingDenseBlocks == 0 {
		t.Fatalf("spec = %+v", spec)
	}
	metadata, tensors, err := DeepSeekOCRConversion(repository)
	if err != nil {
		t.Fatal(err)
	}
	if len(metadata) == 0 || len(tensors) == 0 {
		t.Fatal("conversion produced nothing")
	}
	// The graph that routes among experts takes a block's router in F32, and refuses
	// to be built on any other: a conversion that kept the checkpoint's BF16 wrote a
	// file overgo could not serve.
	routers := uint32(0)
	for _, tensor := range tensors {
		if strings.HasSuffix(tensor.Name, expertRouterSuffix) {
			routers++
			if tensor.Type != gguf.DTypeF32 {
				t.Errorf("%s is written as %v; the routing graph takes it in F32", tensor.Name, tensor.Type)
			}
		}
	}
	if routers != spec.BlockCount-spec.LeadingDenseBlocks {
		t.Fatalf("%d routers written for %d routed blocks", routers, spec.BlockCount-spec.LeadingDenseBlocks)
	}
	t.Logf("blocks=%d experts=%d used=%d shared=%d leading dense=%d tensors=%d", spec.BlockCount, spec.ExpertCount, spec.ExpertUsedCount, spec.SharedExpertCount, spec.LeadingDenseBlocks, len(tensors))
}

// TestDeepSeekOCRProjectorConversionValidatesCheckpoint holds the encoder
// conversion to an mmproj the projector runtime reads: its metadata names the
// two towers, the derived window and the preprocessing, and every tensor the
// runtime requires is present under its name with its shape.
func TestDeepSeekOCRProjectorConversionValidatesCheckpoint(t *testing.T) {
	repository := deepSeekOCRCheckpoint(t)
	metadata, tensors, err := DeepSeekOCRProjectorConversion(repository)
	if err != nil {
		t.Fatal(err)
	}
	window, samBlocks := uint32(0), uint32(0)
	for _, item := range metadata {
		switch item.Key {
		case "clip.vision.window_size":
			window, _ = item.Value.Data.(uint32)
		case "clip.vision.sam.block_count":
			samBlocks, _ = item.Value.Data.(uint32)
		}
	}
	if window == 0 || samBlocks == 0 || len(tensors) == 0 {
		t.Fatalf("projector metadata window=%d sam blocks=%d tensors=%d", window, samBlocks, len(tensors))
	}
	t.Logf("window=%d sam blocks=%d tensors=%d", window, samBlocks, len(tensors))
}
