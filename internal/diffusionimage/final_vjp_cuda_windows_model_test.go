//go:build windows && modeltest

package diffusionimage

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

func TestRealFinalProjectionResidentVJPParity(t *testing.T) {
	cudatest.Require(t)
	model := loadArtifactModel(t)
	golden := loadGolden[struct {
		B, H, W int
		X, Out  []float32
	}](t, "real_forward")
	_, trace, err := model.forward(golden.X, golden.B, golden.H, golden.W, true)
	if err != nil {
		t.Fatal(err)
	}
	dOut := make([]float32, len(golden.Out))
	for index := range dOut {
		dOut[index] = 2 * (golden.Out[index] - golden.X[index]) / float32(len(dOut))
	}
	wantGrads := Grads{}
	wantDX := convTranspose2dStrideBackward(
		wantGrads, model.final.name, trace.finalInput, model.final.weight, dOut,
		golden.B, model.Cfg.BaseChannels, model.Cfg.InChannels,
		trace.finalH, trace.finalW, model.Cfg.PatchSize, model.Cfg.PatchSize,
	)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	program, err := compileFinalProjectionVJP(
		t.Context(), worker, model.final.weight,
		model.Cfg.BaseChannels, model.Cfg.InChannels,
		trace.finalH, trace.finalW, model.Cfg.PatchSize,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer program.Close(t.Context())
	gotDX, gotWeight, err := program.Execute(t.Context(), trace.finalInput, dOut)
	if err != nil {
		t.Fatal(err)
	}
	dxDiff := maxAbsDiff(gotDX, wantDX)
	weightDiff := maxAbsDiff(gotWeight, wantGrads[model.final.name+".weight"])
	for index, value := range append(gotDX, gotWeight...) {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("gradient[%d] is non-finite", index)
		}
	}
	t.Logf("real SimpleDiffusion final VJP: input_max=%.3e weight_max=%.3e input=%d weight=%d", dxDiff, weightDiff, len(gotDX), len(gotWeight))
	if dxDiff > 2e-6 || weightDiff > 2e-6 {
		t.Fatalf("final projection VJP differs: input=%.3e weight=%.3e", dxDiff, weightDiff)
	}
}
