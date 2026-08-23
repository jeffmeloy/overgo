//go:build windows

package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/parity"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestRxBrainProductionParity(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv("OVERGO_CUDA_TEST") != "1" {
		t.Skip("set OVERGO_CUDA_TEST=1 for real RxBrain parity")
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(roots.Models, "Hy-Embodied-RxBrain-1.0")
	image := filepath.Join(model, "Hy-Embodied-RxBrain-1.0", "demo_cases", "bridgev2_move_toy", "input", "obs_1.jpg")
	l := &campaignContext{
		Campaign: parity.NewCampaign(filepath.Join(t.TempDir(), "rxbrain.log")),
		modelDir: model, fixturesDir: filepath.Join(filepath.Dir(roots.Models), "fixtures"),
	}
	if err := verifyActivateVQA(
		l, t.TempDir(), image, "What objects are on the stovetop, and where is the green toy?",
	); err != nil {
		t.Fatal(err)
	}
}
