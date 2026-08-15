//go:build windows && modeltest

package diffusionimage

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

func TestRealDecoderResBlockResidentVJPParity(t *testing.T) {
	cudatest.Require(t)
	model := loadArtifactModel(t)
	golden := loadGolden[struct {
		B, H, W int
		X, Out  []float32
	}](t, "real_forward")
	_, modelTrace, err := model.forward(golden.X, golden.B, golden.H, golden.W, true)
	if err != nil {
		t.Fatal(err)
	}
	levelIndex := len(model.decoders) - 1
	blockIndex := len(model.decoders[levelIndex].blocks) - 1
	block, ok := model.decoders[levelIndex].blocks[blockIndex].(*resBlock)
	if !ok {
		t.Fatal("final decoder block is not residual")
	}
	levelTrace := modelTrace.decLevels[levelIndex]
	input := levelTrace.blockInputs[blockIndex]
	trace := block.forwardTrace(model, input, golden.B, levelTrace.channels, levelTrace.h, levelTrace.width)
	incoming := make([]float32, len(trace.output))
	for index := range incoming {
		incoming[index] = 2 * (trace.output[index] - input[index]) / float32(len(incoming))
	}
	wantGrads := Grads{}
	wantInput := block.backward(model, input, incoming, golden.B, levelTrace.channels, levelTrace.h, levelTrace.width, wantGrads)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	program, err := compileResBlockVJP(t.Context(), worker, block, model.Cfg, levelTrace.channels, levelTrace.h, levelTrace.width)
	if err != nil {
		t.Fatal(err)
	}
	defer program.Close(t.Context())
	gotInput, gotGrads, err := program.Execute(t.Context(), trace, incoming)
	if err != nil {
		t.Fatal(err)
	}
	inputDiff := maxAbsDiff(gotInput, wantInput)
	maxParameterDiff := float64(0)
	for name, want := range wantGrads {
		got, found := gotGrads[name]
		if !found {
			t.Fatalf("missing gradient %s", name)
		}
		difference := maxAbsDiff(got, want)
		maxParameterDiff = math.Max(maxParameterDiff, difference)
		t.Logf("real decoder residual VJP %s max=%.3e elements=%d", name, difference, len(want))
		for index, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("%s gradient[%d] is non-finite", name, index)
			}
		}
	}
	if len(gotGrads) != len(wantGrads) {
		t.Fatalf("gradient set size=%d want=%d", len(gotGrads), len(wantGrads))
	}
	t.Logf("real decoder residual VJP: shape=%dx%dx%d input_max=%.3e parameter_max=%.3e", levelTrace.channels, levelTrace.h, levelTrace.width, inputDiff, maxParameterDiff)
	if inputDiff > 4e-9 || maxParameterDiff > 1e-6 {
		t.Fatalf("decoder residual VJP differs: input=%.3e parameter=%.3e", inputDiff, maxParameterDiff)
	}
}
