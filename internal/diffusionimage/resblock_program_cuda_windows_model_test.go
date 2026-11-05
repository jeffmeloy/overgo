//go:build windows && modeltest

package diffusionimage

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestRealResBlockCUDAParity(t *testing.T) {
	cudatest.Require(t)
	model := loadArtifactModel(t)
	golden := loadGolden[struct {
		B, C, H, W int
		X, Out     []float32
	}](t, "real_forward")
	input, height, width := conv2dValidStride(
		golden.X, model.patch.weight, model.patch.bias,
		golden.B, model.Cfg.InChannels, model.Cfg.BaseChannels,
		golden.H, golden.W, model.Cfg.PatchSize, model.Cfg.PatchSize,
	)
	block, ok := model.encoders[0].blocks[0].(*resBlock)
	if !ok {
		t.Fatal("first encoder block is not residual")
	}
	program, err := compileResBlockProgram(block, model.Cfg, model.Cfg.BaseChannels, height, width)
	testcheck.NoError(t, err)
	want := block.forward(model, input, 1, model.Cfg.BaseChannels, height, width)
	referenceOutput, err := program.execute(context.Background(), nil, input)
	testcheck.NoError(t, err)
	device, err := executor.New(0)
	testcheck.NoError(t, err)
	defer device.Close()
	deviceOutput, err := program.execute(context.Background(), device, input)
	testcheck.NoError(t, err)
	referenceDiff := testutil.MaxAbsDiff(referenceOutput, want)
	deviceDiff := testutil.MaxAbsDiff(deviceOutput, want)
	for index, value := range deviceOutput {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("device output[%d] is non-finite", index)
		}
	}
	t.Logf("real SimpleDiffusion residual block: shape=%dx%dx%d reference_max=%.3e cuda_max=%.3e", model.Cfg.BaseChannels, height, width, referenceDiff, deviceDiff)
	if referenceDiff > 2e-5 || deviceDiff > 2e-3 {
		t.Fatalf("residual block parity exceeds gates: reference=%.3e device=%.3e", referenceDiff, deviceDiff)
	}
}
