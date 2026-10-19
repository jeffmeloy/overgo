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
	verifyMOSSModelStage(t, "refinement")
}

func TestMOSSRotaryJointCheckpoint(t *testing.T) {
	verifyMOSSModelStage(t, "joint")
}

func TestMOSSRotaryDenoiserCheckpoint(t *testing.T) {
	verifyMOSSModelStage(t, "denoiser")
}

func TestMOSSSamplingCheckpoint(t *testing.T) {
	verifyMOSSModelStage(t, "sampling")
}

func TestMOSSDecoderCheckpoint(t *testing.T) {
	verifyMOSSModelStage(t, "decoder")
}

func verifyMOSSModelStage(t *testing.T, stage string) {
	t.Helper()
	cudatest.Require(t)
	modelRoot := testutil.ModelArtifactDir(t, "MOSS-Image-SFT-797fa8e678b4f")
	referenceRoot := os.Getenv("OVERGO_MOSS_REFERENCE_DIR")
	if referenceRoot == "" {
		t.Fatal("OVERGO_MOSS_REFERENCE_DIR must name the frozen MOSS reference directory")
	}
	var report bytes.Buffer
	err := VerifyMOSSTransformer(t.Context(), modelRoot, referenceRoot, filepath.Join("..", "..", "fixtures", "moss"), stage, &report)
	t.Log(report.String())
	if err != nil {
		t.Fatal(err)
	}
}
