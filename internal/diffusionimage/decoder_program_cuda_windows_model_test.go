//go:build windows && modeltest

package diffusionimage

import (
	"fmt"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

func TestRealDecoderBoundariesCUDAParity(t *testing.T) {
	cudatest.Require(t)
	model := loadArtifactModel(t)
	golden := loadGolden[struct {
		B, H, W int
		X       []float32
	}](t, "real_forward")
	if golden.B != 1 {
		t.Fatalf("decoder graph currently requires batch 1, got %d", golden.B)
	}
	_, trace, err := model.forward(golden.X, 1, golden.H, golden.W, true)
	testcheck.NoError(t, err)
	device, err := executor.New(0)
	testcheck.NoError(t, err)
	defer device.Close()
	for index, up := range trace.decUps {
		binding := model.decoders[index].transition
		program, err := compileDecoderTransitionProgram(
			binding,
			imageGeometry{channels: up.channels, height: up.h, width: up.width},
			up.nextChannels,
		)
		testcheck.NoError(t, err)
		want, _, _ := conv2dValidStride(
			up.input, binding.weight, binding.bias, 1,
			up.channels, up.nextChannels, up.h, up.width, 1, 1,
		)
		want = upsampleNearest2x(want, 1, up.nextChannels, up.h, up.width)
		checkProjectionProgram(t, device, fmt.Sprintf("decoder transition %d", index), program, up.input, want)
	}
	finalProgram, err := compileFinalProjectionProgram(
		model.final,
		imageGeometry{channels: model.Cfg.BaseChannels, height: trace.finalH, width: trace.finalW},
		model.Cfg.InChannels, model.Cfg.PatchSize,
	)
	testcheck.NoError(t, err)
	finalWant, _, _ := convTranspose2dStride(
		trace.finalInput, model.final.weight, model.final.bias, 1,
		model.Cfg.BaseChannels, model.Cfg.InChannels, trace.finalH, trace.finalW,
		model.Cfg.PatchSize, model.Cfg.PatchSize,
	)
	checkProjectionProgram(t, device, "final unpatch", finalProgram, trace.finalInput, finalWant)
	t.Logf("real SimpleDiffusion decoder boundaries: transitions=%d output=%dx%d", len(trace.decUps), golden.H, golden.W)
}
