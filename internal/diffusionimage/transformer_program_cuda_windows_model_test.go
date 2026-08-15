//go:build windows && modeltest

package diffusionimage

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
)

func TestRealTransformerBlockCUDAParity(t *testing.T) {
	cudatest.Require(t)
	model := loadArtifactModel(t)
	golden := loadGolden[struct {
		B, H, W int
		X       []float32
	}](t, "real_forward")
	_, trace, err := model.forward(golden.X, golden.B, golden.H, golden.W, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(model.middle) == 0 || len(trace.middleInputs) == 0 {
		t.Fatal("real artifact has no middle transformer block")
	}
	block, input := model.middle[0], trace.middleInputs[0]
	program, err := compileTransformerBlockProgram(block, model.Cfg, trace.middleC, trace.middleH*trace.middleW)
	if err != nil {
		t.Fatal(err)
	}
	want := block.forward(model, input, golden.B, trace.middleC, trace.middleH, trace.middleW)
	referenceOutput, err := program.execute(context.Background(), nil, input)
	if err != nil {
		t.Fatal(err)
	}
	device, err := executor.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	deviceOutput, err := program.execute(context.Background(), device, input)
	if err != nil {
		t.Fatal(err)
	}
	referenceDiff := maxAbsDiff(referenceOutput, want)
	deviceDiff := maxAbsDiff(deviceOutput, want)
	for index, value := range deviceOutput {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("device output[%d] is non-finite", index)
		}
	}
	t.Logf("real SimpleDiffusion transformer block: shape=%dx%d reference_max=%.3e cuda_max=%.3e", trace.middleC, trace.middleH*trace.middleW, referenceDiff, deviceDiff)
	if referenceDiff > 3e-5 || deviceDiff > 2e-3 {
		t.Fatalf("transformer block parity exceeds gates: reference=%.3e device=%.3e", referenceDiff, deviceDiff)
	}
}
