package audiodsp

import (
	"math"
	"slices"
	"testing"
)

func TestFrameConditioningIsLocalAndOwned(t *testing.T) {
	config := testFrontendConfig()
	config.Padding, config.Window = "zero", "rectangular"
	config.FrameSpan, config.WindowOffset = 4, 0
	config.PadLeft, config.PadRight = 0, 0
	config.Condition = &FrameCondition{Gain: 2, RemoveMean: true, Preemphasis: .5}
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	config.Condition.Gain = math.NaN()
	wave := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	var workspace Workspace
	seen := 0
	_, frames, err := plan.Process(t.Context(), [][]float32{wave[:3], wave[3:]}, 8, &workspace, ProcessOptions{Observe: func(trace FrameTrace) error {
		// Every translated ramp has the same centered samples [-3,-1,1,3].
		want := []float64{-1.5, .5, 1.5, 2.5}
		if !slices.Equal(trace.Window, want) {
			t.Fatalf("frame %d conditioned window %v, want %v", trace.Frame, trace.Window, want)
		}
		seen++
		return nil
	}})
	if err != nil || frames != 3 || seen != frames {
		t.Fatalf("conditioning execution %d/%d: %v", seen, frames, err)
	}
	if !slices.Equal(wave, []float32{1, 2, 3, 4, 5, 6, 7, 8}) {
		t.Fatal("modified caller PCM")
	}
	if _, err := plan.Reconstruct(t.Context(), [][]float32{wave}, 8, &workspace); err == nil {
		t.Fatal("claimed reconstruction through irreversible frame centering")
	}
}

func TestDeclaredWindowPowerAndMelInterpolation(t *testing.T) {
	config := testFrontendConfig()
	config.Window = "symmetric-hann"
	exponent := .5
	config.WindowPower = &exponent
	config.SampleRate, config.FFTLength, config.FrameSpan = 4200, 12, 12
	config.Mel.MaxFrequency, config.Mel.LinearInMel = 2100, true
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	exponent = math.NaN()
	if math.Abs(plan.window[1]-math.Sqrt(.75)) > 1e-15 || plan.window[0] != 0 || plan.window[3] != 0 {
		t.Fatalf("powered symmetric Hann %v", plan.window)
	}
	// The 350-Hz bin has normalized mel position log(1.5)/log(4).
	// Its first triangle's descending side is 2-2*log2(1.5).
	want := 2 - 2*math.Log2(1.5)
	if math.Abs(plan.filterbank[plan.bands]-want) > 1e-14 {
		t.Fatalf("mel-linear side %g, want %g", plan.filterbank[plan.bands], want)
	}
	config.WindowPower = nil
	config.Mel.LinearInMel = false
	frequency, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(plan.filterbank, frequency.filterbank) {
		t.Fatal("mel and hertz interpolation were conflated")
	}
}

func TestFixedNormalizationOwnsStatisticsAndBudget(t *testing.T) {
	config := testFrontendConfig()
	plain, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	means, inverse := []float64{1, 2, 3}, []float64{2, 3, 4}
	config.Normalize = &NormalizeConfig{Mode: "fixed", Mean: slices.Clone(means), InverseStd: slices.Clone(inverse)}
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if plan.tableBytes-plain.tableBytes != uint64(len(means)+len(inverse))*float64Bytes {
		t.Fatal("fixed statistics are absent from the retained numeric budget")
	}
	config.Normalize.Mean[0] = math.NaN()
	config.Normalize.InverseStd[0] = math.NaN()
	wave := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	var rawWorkspace, fixedWorkspace Workspace
	raw, _, err := plain.Process(t.Context(), [][]float32{wave}, 8, &rawWorkspace, ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	fixed, _, err := plan.Process(t.Context(), [][]float32{wave}, 8, &fixedWorkspace, ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range raw {
		band := index % len(means)
		want := float32((float64(value) - means[band]) * inverse[band])
		if fixed[index] != want {
			t.Fatalf("fixed normalization[%d]=%g want %g", index, fixed[index], want)
		}
	}
}

func TestConditionDeclarationsRefuseInvalidValues(t *testing.T) {
	for name, mutate := range map[string]func(*FrontendConfig){
		"gain-zero":         func(c *FrontendConfig) { c.Condition = &FrameCondition{} },
		"gain-nan":          func(c *FrontendConfig) { c.Condition = &FrameCondition{Gain: math.NaN()} },
		"coefficient":       func(c *FrontendConfig) { c.Condition = &FrameCondition{Gain: 1, Preemphasis: 2} },
		"window-power":      func(c *FrontendConfig) { value := -1.0; c.WindowPower = &value },
		"rectangular-power": func(c *FrontendConfig) { value := 1.0; c.WindowPower, c.Window = &value, "rectangular" },
		"fixed-shape":       func(c *FrontendConfig) { c.Normalize = &NormalizeConfig{Mode: "fixed"} },
		"fixed-inverse": func(c *FrontendConfig) {
			c.Normalize = &NormalizeConfig{Mode: "fixed", Mean: []float64{0, 0, 0}, InverseStd: []float64{1, 0, 1}}
		},
		"unused-statistics": func(c *FrontendConfig) { c.Normalize = &NormalizeConfig{Mode: "all", Epsilon: 1, Mean: []float64{0}} },
	} {
		t.Run(name, func(t *testing.T) {
			config := testFrontendConfig()
			mutate(&config)
			if _, err := NewFrontend(config, 1<<20); err == nil {
				t.Fatal("accepted invalid declaration")
			}
		})
	}
}
