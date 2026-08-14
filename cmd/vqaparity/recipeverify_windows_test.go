//go:build windows

package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testevidence"
)

func TestRxBrainProductionParity(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv("OVERGO_CUDA_TEST") != "1" {
		t.Skip("set OVERGO_CUDA_TEST=1 for real RxBrain parity")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(roots.Models, "Hy-Embodied-RxBrain-1.0")
	image := filepath.Join(model, "Hy-Embodied-RxBrain-1.0", "demo_cases", "bridgev2_move_toy", "input", "obs_1.jpg")
	l := &ladder{
		modelDir: model, fixturesDir: filepath.Join(filepath.Dir(roots.Models), "fixtures"),
		logPath: filepath.Join(t.TempDir(), "rxbrain.log"),
	}
	if err := verifyActivateVQA(
		l, t.TempDir(), image, "What objects are on the stovetop, and where is the green toy?",
	); err != nil {
		t.Fatal(err)
	}
}
