//go:build windows && modeltest

package diffusionimage

import (
	"context"
	"fmt"
	"math"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
)

func TestAllRealBlocksCUDAParity(t *testing.T) {
	cudatest.Require(t)
	model := loadArtifactModel(t)
	golden := loadGolden[struct {
		B, H, W int
		X       []float32
	}](t, "real_forward")
	if golden.B != 1 {
		t.Fatalf("block graph currently requires batch 1, got %d", golden.B)
	}
	_, trace, err := model.forward(golden.X, 1, golden.H, golden.W, true)
	if err != nil {
		t.Fatal(err)
	}
	device, err := executor.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	count := 0
	checkLevels := func(family string, levels []level, traces []levelTrace) {
		for levelIndex, level := range levels {
			levelTrace := traces[levelIndex]
			for blockIndex, block := range level.blocks {
				name := fmt.Sprintf("%s[%d].block[%d]", family, levelIndex, blockIndex)
				checkRealBlockCUDA(t, device, model, name, block, levelTrace.blockInputs[blockIndex], levelTrace.channels, levelTrace.h, levelTrace.width)
				count++
			}
		}
	}
	checkLevels("encoder", model.encoders, trace.encLevels)
	for index, block := range model.middle {
		checkRealBlockCUDA(t, device, model, fmt.Sprintf("middle[%d]", index), block, trace.middleInputs[index], trace.middleC, trace.middleH, trace.middleW)
		count++
	}
	checkLevels("decoder", model.decoders, trace.decLevels)
	t.Logf("real SimpleDiffusion CUDA block coverage: %d/%d", count, count)
}

func checkRealBlockCUDA(
	t *testing.T,
	device *executor.Executor,
	model *Model,
	name string,
	block block,
	input []float32,
	channels, height, width int,
) {
	t.Helper()
	want := block.forward(model, input, 1, channels, height, width)
	var got []float32
	var err error
	switch typed := block.(type) {
	case *resBlock:
		program, compileErr := compileResBlockProgram(typed, model.Cfg, channels, height, width)
		if compileErr != nil {
			t.Fatal(compileErr)
		}
		got, err = program.execute(context.Background(), device, input)
	case *attnBlock:
		program, compileErr := compileTransformerBlockProgram(typed, model.Cfg, channels, height*width)
		if compileErr != nil {
			t.Fatal(compileErr)
		}
		got, err = program.execute(context.Background(), device, input)
	default:
		t.Fatalf("%s has unsupported block type %T", name, block)
	}
	if err != nil {
		t.Fatal(err)
	}
	difference := maxAbsDiff(got, want)
	for index, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("%s output[%d] is non-finite", name, index)
		}
	}
	t.Logf("%s shape=%dx%dx%d cuda_max=%.3e", name, channels, height, width, difference)
	if difference > 1e-5 {
		t.Fatalf("%s parity exceeds gate: %.3e", name, difference)
	}
}
