package latentimage

import (
	"math"
	"testing"

	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

func TestTimestepProgramMatchesHostReference(t *testing.T) {
	const fixtureSigma = 0.9
	spec := syntheticSpec()
	denoiser := testcheck.Must(NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), syntheticStore(spec)))(t)
	program := testcheck.Must(compileTimestepProgram(spec, dtype.F32, fixtureTimestepProgram(spec)))(t)
	sinusoid := testcheck.Must(program.sinusoid(fixtureSigma))(t)
	feeds := make(map[*tensor.Tensor]reference.Value, len(program.weightInputs)+1)
	feeds[program.Input] = reference.Value{Shape: program.Input.Shape, Data: sinusoid}
	for _, node := range program.weightInputs {
		feeds[node] = reference.Value{Shape: node.Shape, Data: denoiser.w(node.Name)}
	}
	results := testcheck.Must(reference.Execute(
		[]*tensor.Tensor{program.Embedding, program.Modulation}, feeds,
	))(t)
	wantEmbedding, wantModulation, err := denoiser.timestepConditioning(fixtureSigma)
	testcheck.NoError(t, err)
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
