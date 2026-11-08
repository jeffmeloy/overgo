package sampling

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestEverySamplerStepsThroughOneInterface holds overgo's media samplers to one walk: on 2026-11-08 Wan's flow UniPC,
// the image models' flow Euler, RAE's Euler over its float32 grid and SimpleDiffusion's each stepped in a loop of its
// own, so a schedule or a guidance change was written once a loop. A synthetic velocity model walked through Walk
// reaches, bit for bit, what each loop reached stepping by hand, each sampler's deltas rounded as its reference rounds
// them, and the guider combines as both references guide, in float64 rounded once and in float32, its hooks seeing
// the outputs before the combine and the guided one after it.
func TestEverySamplerStepsThroughOneInterface(t *testing.T) {
	t.Parallel()
	const elements = 6
	velocity := func(sigma float64, sample []float32) []float32 {
		output := make([]float32, len(sample))
		for index, value := range sample {
			output[index] = float32(0.5)*value - float32(0.25)*float32(sigma) + float32(index)/100
		}
		return output
	}
	noise := []float32{0.3, -1.2, 0.7, 2.1, -0.4, 0.05}
	walk := func(sampler FlowSampler) []float32 {
		sigmas := sampler.Sigmas()
		return testcheck.Must(Walk(t.Context(), sampler, slices.Clone(noise), func(step int, sample []float32) ([]float32, error) {
			return velocity(sigmas[step], sample), nil
		}))(t)
	}
	byHand := func(sigmas []float64, deltas []float32) []float32 {
		sample := slices.Clone(noise)
		for step, delta := range deltas {
			output := velocity(sigmas[step], sample)
			for index := range sample {
				sample[index] += delta * output[index]
			}
		}
		return sample
	}

	unipcSigmas := testcheck.Must(func() ([]float32, error) {
		_, sigmas, err := UniPCSchedule(1000, 5, 5)
		return sigmas, err
	}())(t)
	unipc := testcheck.Must(NewUniPCSampler(unipcSigmas, elements))(t)
	stepped := testcheck.Must(NewUniPCSampler(unipcSigmas, elements))(t)
	sample, next := slices.Clone(noise), make([]float32, elements)
	for step := range len(unipcSigmas) - 1 {
		testcheck.NoError(t, stepped.Step(next, sample, velocity(float64(unipcSigmas[step]), sample)))
		sample, next = next, sample
	}
	if walked := walk(unipc); !slices.Equal(walked, sample) {
		t.Fatalf("UniPC walked to %v, stepped by hand to %v", walked, sample)
	}

	shifted := []float64{1, 0.8, 0.55, 0.3, 0}
	grid := []float32{1, 0.75, 0.5, 0.25, 0.001}
	tables := map[string]struct {
		sigmas []float64
		deltas []float32
	}{
		"dynamic shift, a float64 change rounded once": {shifted, []float32{float32(0.8 - 1.0), float32(0.55 - 0.8), float32(0.3 - 0.55), float32(0 - 0.3)}},
		"RAE grid, a float32 change":                   {[]float64{1, 0.75, 0.5, 0.25, float64(grid[4])}, []float32{grid[1] - grid[0], grid[2] - grid[1], grid[3] - grid[2], grid[4] - grid[3]}},
		"SimpleDiffusion, one interval":                {[]float64{0, 0.25, 0.5, 0.75, 1}, []float32{0.25, 0.25, 0.25, 0.25}},
	}
	for name, table := range tables {
		euler := testcheck.Must(NewFlowEuler(table.sigmas, table.deltas))(t)
		if walked, want := walk(euler), byHand(table.sigmas, table.deltas); !slices.Equal(walked, want) {
			t.Fatalf("%s: Euler walked to %v, stepped by hand to %v", name, walked, want)
		}
		if err := euler.Step(make([]float32, elements), noise, noise); err == nil {
			t.Fatalf("%s: an Euler walk stepped past its table", name)
		}
	}

	conditional, unconditional := []float32{1.5, -0.2, 3}, []float32{0.5, 0.4, -1}
	var before, after int
	guider := Guider{Scale: 5, Before: func([]float32, []float32) error { before++; return nil }, After: func([]float32) error { after++; return nil }}
	guided, reference := make([]float32, 3), make([]float32, 3)
	testcheck.NoError(t, guider.Combine(guided, conditional, unconditional))
	testcheck.NoError(t, GuideInto(reference, conditional, unconditional, 5))
	single := slices.Clone(conditional)
	testcheck.NoError(t, Guider{Scale: 5, Float32: true}.Combine(single, single, unconditional))
	for index, value := range unconditional {
		if want := value + float32(5)*(conditional[index]-value); single[index] != want {
			t.Fatalf("a float32 guider gave %v at %d, its reference %v", single[index], index, want)
		}
	}
	if !slices.Equal(guided, reference) || before != 1 || after != 1 {
		t.Fatalf("the guider gave %v against %v, its hooks run %d and %d times", guided, reference, before, after)
	}
}
