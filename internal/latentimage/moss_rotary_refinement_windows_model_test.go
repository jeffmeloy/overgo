//go:build windows && modeltest

package latentimage

import (
	"bytes"
	"os"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
	"path/filepath"
	"testing"
)

func TestMOSSRotaryRefinementCheckpoint(t *testing.T) {
	cudatest.Require(t)
	modelRoot := testutil.ModelArtifactDir(t, "MOSS-Image-SFT-797fa8e678b4f")
	referenceRoot := os.Getenv("OVERGO_MOSS_REFERENCE_DIR")
	if referenceRoot == "" {
		t.Fatal("OVERGO_MOSS_REFERENCE_DIR must name the frozen MOSS reference directory")
	}
	var report bytes.Buffer
	err := VerifyMOSSRefinement(t.Context(), modelRoot, referenceRoot, filepath.Join("..", "..", "fixtures", "moss"), &report)
	t.Log(report.String())
	if err != nil {
		t.Fatal(err)
	}
}
