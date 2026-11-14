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
	const gh, gw, textSeq, sigma = 2, 2, 3, 0.9
	spec := syntheticSpec()
	denoiser := testcheck.Must(NewDenoiser(spec, 1e-5, fixtureTimestepProgram(spec), syntheticStore(spec)))(t)
	latent, encoded := syntheticForwardInputs(spec, textSeq, gh*gw)
	patches := mustPack(t, latent, spec.InChannels, gh, gw)
	mask := []bool{true, true, false}
	unguided := testcheck.Must(testcheck.Must(CompileDenoiserProgram(spec, 1e-5, mask, nil, gh, gw, dtype.F32))(t).Forward(reference.Execute, denoiser, patches, encoded, sigma))(t).Velocity
	prompt := testcheck.Must(denoiser.textConditioning(encoded, textSeq))(t)
	guide := NegativeGuide{Mask: mask, Attention: sampling.NormalizedAttention{Scale: 4, Tau: 2.5, Alpha: 0.25}}
	program := testcheck.Must(CompileDenoiserProgram(spec, 1e-5, mask, nil, gh, gw, dtype.F32, guide))(t)
	velocity := func(negative []float64) []float32 {
		t.Helper()
		feeds := testcheck.Must(program.hostFeeds(denoiser, patches, encoded, sigma))(t)
		feeds[program.InNegative] = reference.Value{Shape: program.InNegative.Shape, Data: dtype.Float64SliceToFloat32(negative)}
		feeds[program.negativeBias] = reference.Value{Shape: program.negativeBias.Shape, Data: program.negativeData}
		full := testcheck.Must(reference.Execute([]*tensor.Tensor{program.Velocity}, feeds))(t)[program.Velocity].Data
		return full[textSeq*spec.InChannels:]
	}
	// apart is the largest difference of two velocities, each value held finite first: a NaN would pass both bounds,
	// since max carries it and no comparison with it holds (SQA-402).
	apart := func(a, b []float32) float64 {
		largest := 0.0
		for index := range a {
			if !checked.Finite32(a[index]) || !checked.Finite32(b[index]) {
				t.Fatalf("velocity %d is %g against %g, not finite", index, a[index], b[index])
			}
			largest = max(largest, math.Abs(float64(a[index]-b[index])))
		}
		return largest
	}
	if gap := apart(velocity(prompt), unguided); gap > 1e-6 {
		t.Fatalf("the prompt as its own negative moved the velocity by %.3e; NAG of Z+ against itself is Z+", gap)
	}
	other := slices.Clone(prompt)
	for index := range other {
		other[index] = math.Sin(float64(index) * 0.23)
	}
	moved := velocity(other)
	if gap := apart(moved, unguided); gap < 1e-4 {
		t.Fatalf("another negative prompt moved the velocity by %.3e", gap)
	}
	padded := slices.Clone(other)
	for index := range spec.Hidden {
		padded[2*spec.Hidden+index] += 3
	}
	if !slices.Equal(velocity(padded), moved) {
		t.Fatal("a change to the negative prompt's padding row, which its mask leaves unattended, moved the velocity")
	}
}
