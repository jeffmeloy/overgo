package latentimage

import (
	"math"
	"testing"

	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestTimestepProgramMatchesHostReference(t *testing.T) {
	const fixtureSigma = 0.9
	spec := syntheticSpec()
	denoiser, err := NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), syntheticStore(spec))
	if err != nil {
		t.Fatal(err)
	}
	program, err := compileTimestepProgram(spec, dtype.F32, fixtureTimestepProgram(spec))
	if err != nil {
		t.Fatal(err)
	}
	sinusoid, err := program.sinusoid(fixtureSigma)
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(program.weightInputs)+1)
	feeds[program.Input] = reference.Value{Shape: program.Input.Shape, Data: sinusoid}
	for name, node := range program.weightInputs {
		feeds[node] = reference.Value{Shape: node.Shape, Data: denoiser.w(name)}
	}
	results, err := reference.Execute(
		[]*tensor.Tensor{program.Embedding, program.Modulation}, feeds,
	)
	if err != nil {
		t.Fatal(err)
	}
	wantEmbedding, wantModulation, err := denoiser.timestepConditioning(fixtureSigma)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat64Near(t, results[program.Embedding].Data, wantEmbedding)
	assertFloat64Near(t, results[program.Modulation].Data, wantModulation)
}

func fixtureTimestepProgram(spec TransformerSpec) media.SinusoidalProgram {
	return media.SinusoidalProgram{Dimensions: spec.TimestepEmbed, FrequencyBase: 1e4, InputScale: 1e3}
}

func assertFloat64Near(t *testing.T, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	const tolerance = 1e-4
	for index := range got {
		if delta := math.Abs(float64(got[index]) - want[index]); delta > tolerance {
			t.Fatalf("value[%d] = %g, want %g (delta %g)", index, got[index], want[index], delta)
		}
	}
}
