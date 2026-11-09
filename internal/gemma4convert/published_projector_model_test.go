//go:build modeltest

package gemma4convert_test

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/projector"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// publishedGemma is the checkpoint the second judge's vision is read from, as its publisher released it.
const publishedGemma = "gemma-4-12B-it"

// TestGemmaTwelvesProjectorOpensFromItsPublishedDirectory: gemma-4-12B-it's media projector opens from the checkpoint as
// published, by the names its model.safetensors gives its embedder (model.vision_embedder.*,
// model.embed_vision.embedding_projection), which the converter's mapping did not name, so neither the view nor
// convert-gemma4 opened it (2026-11-09). Its spec is the checkpoint's: a 16-pixel patch pooled by 3, 3840 wide, 1120
// positions; it embeds a picture on the host, and nothing is written beside the checkpoint.
func TestGemmaTwelvesProjectorOpensFromItsPublishedDirectory(t *testing.T) {
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	directory := roots.ResolveModelPath(publishedGemma)
	if _, err := os.Stat(filepath.Join(directory, "model.safetensors")); err != nil {
		testskip.NotApplicable(t, publishedGemma+" is not on this machine: "+err.Error())
	}
	published := testcheck.Must(os.ReadDir(directory))(t)
	runner := testcheck.Must(projector.OpenAs[*projector.Gemma4Runner](t.Context(), directory, projector.OpenOptions{}))(t)
	defer runner.Close()
	spec := runner.Spec()
	if spec.TeacherPatch != 16 || spec.PoolKernel != 3 || spec.Hidden != 3840 || spec.PositionCount != 1120 {
		t.Fatalf("%s's projector has spec %+v, want a 16-pixel patch pooled by 3, 3840 wide, 1120 positions", publishedGemma, spec)
	}
	picture := image.NewRGBA(image.Rect(0, 0, 96, 96))
	for index := range picture.Pix {
		picture.Pix[index] = uint8(index)
	}
	output := testcheck.Must(runner.EncodeImage(t.Context(), picture))(t)
	if len(output.Embeddings.Data) == 0 || len(output.Embeddings.Data)%spec.Hidden != 0 {
		t.Fatalf("%s's projector embedded a picture as %d values, want whole rows of %d", publishedGemma, len(output.Embeddings.Data), spec.Hidden)
	}
	if after := testcheck.Must(os.ReadDir(directory))(t); len(after) != len(published) {
		t.Fatalf("opening the projector left %d entries in %s, want the %d published", len(after), directory, len(published))
	}
}
