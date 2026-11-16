package latentimage

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/checked"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestKreasNAGGuidesTheImageRowsByTheNegativePrompt holds NAG in Krea's co-attention as guidance-ops/nag-krea reads the
// paper for joint attention, on the host reference: the negative prompt's rows run as a stream of their own beside the
// one image stream, so a negative prompt that is the prompt makes Z- what Z+ is, NAG's combination gives Z+ back, and
// the guided step's velocity is the unguided one's to f32 rounding; another negative prompt moves it; and a change to
// a padding row of the negative prompt, whose key its mask leaves unattended, moves nothing.
func TestKreasNAGGuidesTheImageRowsByTheNegativePrompt(t *testing.T) {
	t.Parallel()
	mask := []bool{true, true, false}
	guided := newGuidedStep(t, mask, nil)
	unguided := guided.unguided(t)
	program := guided.compile(t, NegativeGuide{Mask: mask, Attention: guided.attention})
	if gap := velocityGap(t, guided.velocity(t, program, guided.prompt), unguided); gap > 1e-6 {
		t.Fatalf("the prompt as its own negative moved the velocity by %.3e; NAG of Z+ against itself is Z+", gap)
	}
	other := slices.Clone(guided.prompt)
	for index := range other {
		other[index] = math.Sin(float64(index) * 0.23)
	}
	moved := guided.velocity(t, program, other)
	if gap := velocityGap(t, moved, unguided); gap < 1e-4 {
		t.Fatalf("another negative prompt moved the velocity by %.3e", gap)
	}
	padded := slices.Clone(other)
	for index := range guided.spec.Hidden {
		padded[2*guided.spec.Hidden+index] += 3
	}
	if !slices.Equal(guided.velocity(t, program, padded), moved) {
		t.Fatal("a change to the negative prompt's padding row, which its mask leaves unattended, moved the velocity")
	}
}

// guidedStep is one Krea step on the host reference, a prompt of three rows over a 2x2 picture, under a mask and the
// prompt's weights, and the attention guidance a negative prompt guides it by.
type guidedStep struct {
	spec      TransformerSpec
	denoiser  *Denoiser
	patches   []float64
	encoded   []float64
	prompt    []float64
	mask      []bool
	weights   []float32
	attention sampling.NormalizedAttention
}

const guidedRows, guidedHeight, guidedWidth, guidedSigma = 3, 2, 2, 0.9

func newGuidedStep(t *testing.T, mask []bool, weights []float32) guidedStep {
	t.Helper()
	spec := syntheticSpec()
	denoiser := testcheck.Must(NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), syntheticStore(spec)))(t)
	latent, encoded := syntheticForwardInputs(spec, guidedRows, guidedHeight*guidedWidth)
	return guidedStep{
		spec: spec, denoiser: denoiser, patches: mustPack(t, latent, spec.InChannels, guidedHeight, guidedWidth), encoded: encoded,
		prompt: testcheck.Must(denoiser.textConditioning(encoded, guidedRows))(t), mask: mask, weights: weights,
		attention: sampling.NormalizedAttention{Scale: 4, Tau: 2.5, Alpha: 0.25},
	}
}

// compile is the step's program, guided by the guides given and unguided by none.
func (s guidedStep) compile(t *testing.T, guides ...NegativeGuide) *DenoiserProgram {
	t.Helper()
	return testcheck.Must(CompileDenoiserProgram(s.spec, 1e-5, s.mask, s.weights, guidedHeight, guidedWidth, dtype.F32, guides...))(t)
}

// unguided is the image rows' velocity of the step guided by nothing.
func (s guidedStep) unguided(t *testing.T) []float32 {
	t.Helper()
	return testcheck.Must(s.compile(t).Forward(reference.Execute, s.denoiser, s.patches, s.encoded, guidedSigma))(t).Velocity
}

// velocity is the image rows' velocity of a guided program given the negative prompt's conditioning.
func (s guidedStep) velocity(t *testing.T, program *DenoiserProgram, negative []float64) []float32 {
	t.Helper()
	feeds := testcheck.Must(program.hostFeeds(s.denoiser, s.patches, s.encoded, guidedSigma))(t)
	feeds[program.InNegative] = reference.Value{Shape: program.InNegative.Shape, Data: dtype.Float64SliceToFloat32(negative)}
	full := testcheck.Must(reference.Execute([]*tensor.Tensor{program.Velocity}, feeds))(t)[program.Velocity].Data
	return full[guidedRows*s.spec.InChannels:]
}

// velocityGap is the largest difference of two velocities, each value held finite first: a NaN would pass both bounds,
// since max carries it and no comparison with it holds (SQA-402).
func velocityGap(t *testing.T, a, b []float32) float64 {
	t.Helper()
	largest := 0.0
	for index := range a {
		if !checked.Finite32(a[index]) || !checked.Finite32(b[index]) {
			t.Fatalf("velocity %d is %g against %g, not finite", index, a[index], b[index])
		}
		largest = max(largest, math.Abs(float64(a[index]-b[index])))
	}
	return largest
}
