package audiodsp

import (
	"math"
	"slices"
	"testing"
)

func TestDeclaredNormalizationPrecision(t *testing.T) {
	// The middle addition is lost in a sequential float32 sum, but not in
	// float64. This fixture tests the arithmetic contract, not a model tolerance.
	for _, precision := range []bool{false, true} {
		p := &Frontend{bands: 1, config: FrontendConfig{Log: LogConfig{Scale: 1}, Normalize: &NormalizeConfig{Mode: "all", Epsilon: 1, Float32: precision}}}
		values := []float32{1 << 24, 1, -1 << 24}
		if err := p.transform(t.Context(), values, 3); err != nil {
			t.Fatal(err)
		}
		if precision {
			if values[0] != -values[2] || values[1] <= 0 {
				t.Fatalf("float32 statistics: %v", values)
			}
		} else if values[1] >= float32(1/math.Sqrt(float64(2*(1<<48))/3)) {
			t.Fatalf("float64 mean was rounded to zero: %v", values)
		}
	}
	for _, epsilon := range []float64{math.SmallestNonzeroFloat64, math.MaxFloat64} {
		if err := (NormalizeConfig{Mode: "all", Epsilon: epsilon, Float32: true}).validate(1); err == nil {
			t.Fatal("invalid float32 epsilon accepted")
		}
	}
	if err := (NormalizeConfig{Mode: "fixed", Float32: true, Mean: []float64{0}, InverseStd: []float64{1}}).validate(1); err == nil {
		t.Fatal("float32 fixed statistics accepted")
	}
}

func TestDeclaredPreemphasisPrecision(t *testing.T) {
	c := testFrontendConfig()
	c.Padding = "zero"
	c.Window = "rectangular"
	c.FrameSpan = 4
	c.WindowOffset = 0
	c.PadLeft = 0
	c.PadRight = 0
	coefficient := .97
	c.WaveformPreemphasis = &coefficient
	c.PreemphasisFloat32 = true
	p, err := NewFrontend(c, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	wave := []float32{.3, .1, .2, .4}
	original := slices.Clone(wave)
	want := []float64{float64(wave[0]), float64(wave[1] - float32(float32(coefficient)*wave[0])), float64(wave[2] - float32(float32(coefficient)*wave[1])), float64(wave[3] - float32(float32(coefficient)*wave[2]))}
	var w Workspace
	_, _, err = p.Process(t.Context(), [][]float32{wave[:1], wave[1:]}, c.SampleRate, &w, ProcessOptions{Observe: func(trace FrameTrace) error {
		if !slices.Equal(trace.Window, want) {
			t.Fatalf("window %v want %v", trace.Window, want)
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(wave, original) {
		t.Fatal("modified input")
	}
	c.WaveformPreemphasis = nil
	if _, err := NewFrontend(c, 1<<20); err == nil {
		t.Fatal("precision without operation accepted")
	}
}
