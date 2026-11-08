//go:build windows && modeltest

package latentimage

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
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

// TestMOSSGenerationCheckpoint holds whole generations to the pinned upstream source.
func TestMOSSGenerationCheckpoint(t *testing.T) {
	verifyMOSSModelStage(t, "generation")
}

// TestMOSSEncoderCheckpoint holds the native encoder to the pinned diffusers source
// on the real checkpoint.
func TestMOSSEncoderCheckpoint(t *testing.T) {
	cudatest.Require(t)
	encoderGoldenCheck(t, testutil.ModelArtifactDir(t, "MOSS-Image-SFT-797fa8e678b4f"), "encoder-model.safetensors", "2b1ef4be74a3a16ca30e95d9fff674301f8a1273ec2ce38a9b34f90553cf4f63", 64, 96, 32)
}

func verifyMOSSModelStage(t *testing.T, stage string) {
	t.Helper()
	cudatest.Require(t)
	modelRoot := testutil.ModelArtifactDir(t, "MOSS-Image-SFT-797fa8e678b4f")
	// The frozen reference the checkpoint's goldens were drawn from stands beside the store, as every reference does.
	referenceRoot := filepath.Join(testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t).References(), "moss-image", "goldens-797fa8e678b4f")
	if _, err := os.Stat(referenceRoot); err != nil {
		testskip.NotApplicable(t, "no frozen MOSS reference is at "+referenceRoot)
	}
	var report bytes.Buffer
	err := VerifyMOSSTransformer(t.Context(), modelRoot, referenceRoot, filepath.Join("..", "..", "fixtures", "moss"), stage, &report)
	t.Log(report.String())
	testcheck.NoError(t, err)
}
