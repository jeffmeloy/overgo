//go:build windows && modeltest

package diffusionimage

import (
	"context"
	"fmt"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
)

func TestRealProjectionBoundariesCUDAParity(t *testing.T) {
	cudatest.Require(t)
	model := loadArtifactModel(t)
	golden := loadGolden[struct {
		B, H, W int
		X       []float32
	}](t, "real_forward")
	if golden.B != 1 {
		t.Fatalf("projection graph currently requires batch 1, got %d", golden.B)
	}
	device, err := executor.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()

	patch, err := compileProjectionProgram(
		model.patch,
		imageGeometry{channels: model.Cfg.InChannels, height: golden.H, width: golden.W},
		model.Cfg.BaseChannels, model.Cfg.PatchSize, model.Cfg.PatchSize, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	patchWant, patchH, patchW := conv2dValidStride(
		golden.X, model.patch.weight, model.patch.bias, 1,
		model.Cfg.InChannels, model.Cfg.BaseChannels, golden.H, golden.W,
		model.Cfg.PatchSize, model.Cfg.PatchSize,
	)
	checkProjectionProgram(t, device, "patch", patch, golden.X, patchWant)

	_, trace, err := model.forward(golden.X, 1, golden.H, golden.W, true)
	if err != nil {
		t.Fatal(err)
	}
	for index, down := range trace.encDowns {
		binding := model.encoders[index].transition
		downProgram, err := compileProjectionProgram(
			binding,
			imageGeometry{channels: down.channels, height: down.h, width: down.width},
			down.nextChannels, 1, 1, true,
		)
		if err != nil {
			t.Fatal(err)
		}
		downWant, _, _ := conv2dValidStride(
			down.input, binding.weight, binding.bias, 1,
			down.channels, down.nextChannels, down.h, down.width, 1, 1,
		)
		downWant = avgPool2x(downWant, 1, down.nextChannels, down.h, down.width)
		checkProjectionProgram(t, device, fmt.Sprintf("encoder transition %d", index), downProgram, down.input, downWant)
		t.Logf("encoder transition %d: %dx%dx%d", index, down.nextChannels, down.h/2, down.width/2)
	}
	t.Logf("real SimpleDiffusion patch: %dx%dx%d", model.Cfg.BaseChannels, patchH, patchW)
}

func checkProjectionProgram(t *testing.T, device *executor.Executor, name string, program projectionProgram, input, want []float32) {
	t.Helper()
	referenceOutput, err := program.execute(context.Background(), nil, input)
	if err != nil {
		t.Fatal(err)
	}
	deviceOutput, err := program.execute(context.Background(), device, input)
	if err != nil {
		t.Fatal(err)
	}
	referenceDiff, deviceDiff := maxAbsDiff(referenceOutput, want), maxAbsDiff(deviceOutput, want)
	t.Logf("%s reference_max=%.3e cuda_max=%.3e", name, referenceDiff, deviceDiff)
	if referenceDiff > 2e-5 || deviceDiff > 2e-3 {
		t.Fatalf("%s parity exceeds gates: reference=%.3e device=%.3e", name, referenceDiff, deviceDiff)
	}
}
