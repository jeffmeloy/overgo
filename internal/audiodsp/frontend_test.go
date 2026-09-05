package audiodsp

import (
	"context"
	"math"
	"testing"

	"overgo/internal/recipecontract"
)

func testFrontendConfig() FrontendConfig {
	return FrontendConfig{
		SampleRate: 8,
		Geometry:   recipecontract.AudioFrameGeometry{WindowSamples: 4, HopSamples: 2, FeatureBins: 3},
		FFTLength:  8, FrameSpan: 8, WindowOffset: 2, PadLeft: 4, PadRight: 4,
		Padding: "reflect", Window: "periodic-hann",
		Mel: MelConfig{Scale: "htk", MinFrequency: 0, MaxFrequency: 4},
		Log: LogConfig{Power: true, Base: "natural", GuardMode: "add", Guard: math.SmallestNonzeroFloat64, Scale: 1},
	}
}

func TestSpectrumMatchesNumpyRFFT(t *testing.T) {
	plan, err := NewFrontend(testFrontendConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	samples := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	var workspace Workspace
	source, frames, err := plan.prepare(t.Context(), [][]float32{samples[:1], samples[1:6], samples[6:]}, 8, &workspace)
	if err != nil {
		t.Fatal(err)
	}
	if frames != 5 {
		t.Fatalf("frames=%d want 5", frames)
	}
	if err := plan.workspace(&workspace, 0, len(samples), false); err != nil {
		t.Fatal(err)
	}
	wantReal := [][]float64{{3, -2.414213562373095, 1, .41421356237309515, -1}, {6, -5.121320343559643, 3, -.8786796564403572, 0}, {0, .2928932188134524, -1, 1.7071067811865475, -2}, {-6, 5.121320343559643, -3, .8786796564403572, 0}, {-6, 5.121320343559643, -3, .8786796564403572, 0}}
	wantImaginary := [][]float64{{0, 0, 0, 0, 0}, {0, .7071067811865476, -1, .7071067811865476, 0}, {0, -2.121320343559643, 3, -2.121320343559643, 0}, {0, -.7071067811865476, 1, -.7071067811865476, 0}, {0, .7071067811865476, -1, .7071067811865476, 0}}
	for frame := range frames {
		plan.spectrum(&source, frame, &workspace)
		for bin := range plan.bins {
			if math.Abs(workspace.real[bin]-wantReal[frame][bin]) > 1e-14 || math.Abs(workspace.imaginary[bin]-wantImaginary[frame][bin]) > 1e-14 {
				t.Fatalf("frame %d bin %d=(%.17g,%.17g) want (%.17g,%.17g)", frame, bin, workspace.real[bin], workspace.imaginary[bin], wantReal[frame][bin], wantImaginary[frame][bin])
			}
		}
	}
}

func TestProcessIsChunkBoundaryInvariantAndReusesStorage(t *testing.T) {
	plan, err := NewFrontend(testFrontendConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	samples := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	var whole, split Workspace
	want, frames, err := plan.Process(t.Context(), [][]float32{samples}, 8, &whole, ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	for boundary := 1; boundary < len(samples); boundary++ {
		got, gotFrames, err := plan.Process(t.Context(), [][]float32{samples[:boundary], samples[boundary:]}, 8, &split, ProcessOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if gotFrames != frames || len(got) != len(want) {
			t.Fatalf("boundary %d geometry=%d/%d", boundary, gotFrames, len(got))
		}
		for index := range want {
			if got[index] != want[index] {
				t.Fatalf("boundary %d value %d=%g want %g", boundary, index, got[index], want[index])
			}
		}
	}
	first := &split.features[0]
	if _, _, err := plan.Process(t.Context(), [][]float32{samples[:3], samples[3:]}, 8, &split, ProcessOptions{}); err != nil {
		t.Fatal(err)
	}
	if first != &split.features[0] {
		t.Fatal("feature backing storage was not reused")
	}
}

func TestStandardSTFTInverseReconstructsSource(t *testing.T) {
	config := testFrontendConfig()
	config.Geometry.WindowSamples = 8
	config.Geometry.HopSamples = 4
	config.FrameSpan = 8
	config.WindowOffset = 0
	config.Window = "rectangular"
	config.PadLeft = 4
	config.PadRight = 4
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	samples := []float32{.125, -.25, .5, -.75, 1, -.625, .375, -.125, .0625}
	var workspace Workspace
	got, err := plan.Reconstruct(t.Context(), [][]float32{samples[:2], samples[2:7], samples[7:]}, 8, &workspace)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(samples) {
		t.Fatalf("samples=%d want %d", len(got), len(samples))
	}
	for index, want := range samples {
		if math.Abs(float64(got[index]-want)) > 4*math.SmallestNonzeroFloat32 {
			t.Fatalf("sample %d=%g want %g", index, got[index], want)
		}
	}
}

func TestNormalizationUsesOnlyDeclaredStatistics(t *testing.T) {
	config := testFrontendConfig()
	config.Padding = "zero"
	config.PadLeft = 0
	config.PadRight = 0
	config.FrameSpan = 4
	config.WindowOffset = 0
	config.FFTLength = 4
	config.Window = "rectangular"
	config.Geometry = recipecontract.AudioFrameGeometry{WindowSamples: 4, HopSamples: 4, FeatureBins: 2}
	config.Mel.MaxFrequency = 4
	config.Normalize = &NormalizeConfig{Mode: "per-feature", Correction: 0, Epsilon: 1}
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var workspace Workspace
	got, frames, err := plan.Process(t.Context(), [][]float32{{1, 2, 3, 4, 4, 3, 2, 1}}, 8, &workspace, ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if frames != 2 || len(got) != 4 {
		t.Fatalf("geometry=%d/%d", frames, len(got))
	}
	for band := range 2 {
		if math.Abs(float64(got[band]+got[2+band])) > 1e-7 {
			t.Fatalf("band %d is not centered: %v", band, got)
		}
	}
}

func TestFrontendRejectsBorrowedWorkspaceInput(t *testing.T) {
	p, err := NewFrontend(testFrontendConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var w Workspace
	values, _, err := p.Process(t.Context(), [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}, 8, &w, ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got, _, err := p.Process(t.Context(), [][]float32{values[1:]}, 8, &w, ProcessOptions{}); err == nil || got != nil {
		t.Fatal("accepted borrowed output as mutable input")
	}
}

func TestFrontendSnapshotsRecipeParameters(t *testing.T) {
	config := testFrontendConfig()
	limit := 8.0
	config.Log.DynamicRange = &limit
	config.Normalize = &NormalizeConfig{Mode: "all", Correction: 0, Epsilon: 1}
	config.ResampleTaps = []float64{1}
	p, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	limit = 0
	config.Normalize.Epsilon = 0
	config.ResampleTaps[0] = math.NaN()
	if *p.config.Log.DynamicRange != 8 || p.config.Normalize.Epsilon != 1 || p.config.ResampleTaps[0] != 1 {
		t.Fatal("plan retained caller-owned configuration storage")
	}
}

func TestFrontendSlaneyFilterbankMatchesTorchAudio(t *testing.T) {
	config := testFrontendConfig()
	config.SampleRate = 8000
	config.Mel = MelConfig{Scale: "slaney", MaxFrequency: 4000, AreaNormalize: true}
	p, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	// torchaudio.functional.melscale_fbanks(5,0,4000,3,8000,
	// norm="slaney",mel_scale="slaney"), computed by the installed CPU oracle.
	want := []float64{0, 0, 0, .0005348663544282317, .0008510092739015818, 0, 0, .00023411086294800043, .0005793526652269065, 0, 0, .00039287295658141375, 0, 0, 0}
	// The oracle stores weights in float32; this reference computes float64.
	for index, value := range want {
		if math.Abs(p.filterbank[index]-value) > 8*(1.0/(1<<23))*max(1e-3, value) {
			t.Fatalf("weight %d=%.17g want %.17g", index, p.filterbank[index], value)
		}
	}
}

func TestFrontendNormalizationAndDynamicRangeDefinitions(t *testing.T) {
	config := testFrontendConfig()
	config.Geometry.FeatureBins = 2
	config.Normalize = &NormalizeConfig{Mode: "per-feature", Correction: 0, Epsilon: 1}
	p, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	values := []float32{1, 10, 3, 30}
	if err := p.transform(t.Context(), values, 2); err != nil {
		t.Fatal(err)
	}
	want := []float32{-.5, -10.0 / 11, .5, 10.0 / 11}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("normalization=%v want=%v", values, want)
		}
	}
	config.Normalize = nil
	dynamicRange := 8.0
	config.Log.DynamicRange = &dynamicRange
	config.Log.Scale = .25
	config.Log.Bias = 1
	p, err = NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	values = []float32{-10, 0}
	if err := p.transform(t.Context(), values, 1); err != nil {
		t.Fatal(err)
	}
	if values[0] != -1 || values[1] != 1 {
		t.Fatalf("log range=%v", values)
	}
}

func FuzzFrontend(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	f.Add([]byte{255, 0, 255, 127, 128})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 2 || len(data) > 128 {
			return
		}
		config := testFrontendConfig()
		config.Padding = "zero"
		config.Geometry.WindowSamples = uint64(1 + data[0]%8)
		config.FFTLength = 8
		config.FrameSpan = 8
		config.WindowOffset = (8 - int(config.Geometry.WindowSamples)) / 2
		config.Geometry.HopSamples = uint64(1 + data[1]%8)
		if data[0]%2 == 0 {
			config.Padding = "reflect"
		}
		p, err := NewFrontend(config, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		samples := make([]float32, len(data))
		for i, v := range data {
			samples[i] = float32(int(v)-128) / 128
		}
		boundary := len(samples) / 2
		chunks := [][]float32{nil, samples[:boundary], nil, samples[boundary:], nil}
		var w Workspace
		values, frames, err := p.Process(t.Context(), chunks, 8, &w, ProcessOptions{})
		if err == nil {
			if len(values) != frames*p.bands {
				t.Fatal("invalid feature geometry")
			}
			for _, v := range values {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatal("non-finite feature")
				}
			}
		} else if values != nil {
			t.Fatal("partial features on error")
		}
		wave, err := p.Reconstruct(t.Context(), chunks, 8, &w)
		if err == nil {
			if len(wave) != len(samples) {
				t.Fatal("invalid inverse length")
			}
		} else if wave != nil {
			t.Fatal("partial inverse on error")
		}
	})
}

func TestFrontendSharedPlanSeparateWorkspaces(t *testing.T) {
	plan, err := NewFrontend(testFrontendConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	samples := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	for _, operation := range []string{"features", "inverse"} {
		t.Run(operation, func(t *testing.T) {
			t.Parallel()
			var workspace Workspace
			if operation == "features" {
				values, frames, err := plan.Process(t.Context(), [][]float32{samples}, 8, &workspace, ProcessOptions{})
				if err != nil || frames != 5 || len(values) != 15 {
					t.Fatalf("features=%d/%d error=%v", frames, len(values), err)
				}
			} else {
				values, err := plan.Reconstruct(t.Context(), [][]float32{samples[:3], samples[3:]}, 8, &workspace)
				if err != nil || len(values) != len(samples) {
					t.Fatalf("inverse=%d error=%v", len(values), err)
				}
			}
		})
	}
}

func TestResamplingUsesDeclaredFIRAndTargetGrid(t *testing.T) {
	config := testFrontendConfig()
	config.SampleRate = 2
	config.Padding = "zero"
	config.PadLeft = 0
	config.PadRight = 0
	config.FrameSpan = 2
	config.FFTLength = 2
	config.WindowOffset = 0
	config.Window = "rectangular"
	config.Geometry = recipecontract.AudioFrameGeometry{WindowSamples: 2, HopSamples: 2, FeatureBins: 1}
	config.Mel.MaxFrequency = 1
	config.ResampleTaps = []float64{0, 1, 0}
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var workspace Workspace
	_, frames, err := plan.Process(t.Context(), [][]float32{{1}, {2}}, 1, &workspace, ProcessOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if frames != 2 || len(workspace.resampled) != 4 {
		t.Fatalf("resampled geometry=%d/%d", frames, len(workspace.resampled))
	}
	want := []float32{1, 0, 2, 0}
	for i := range want {
		if workspace.resampled[i] != want[i] {
			t.Fatalf("resampled=%v", workspace.resampled)
		}
	}
}

func TestFrontendRefusesImplicitOrUnboundedBehavior(t *testing.T) {
	base := testFrontendConfig()
	cases := []struct {
		name   string
		mutate func(*FrontendConfig)
		budget uint64
	}{
		{"zero budget", func(*FrontendConfig) {}, 0},
		{"unknown padding", func(c *FrontendConfig) { c.Padding = "replicate" }, 1 << 20},
		{"window outside FFT", func(c *FrontendConfig) { c.FFTLength = 3 }, 1 << 20},
		{"Nyquist violation", func(c *FrontendConfig) { c.Mel.MaxFrequency = 5 }, 1 << 20},
		{"even FIR", func(c *FrontendConfig) { c.ResampleTaps = []float64{1, 1} }, 1 << 20},
		{"unspecified log scale", func(c *FrontendConfig) { c.Log.Scale = math.NaN() }, 1 << 20},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			config := base
			test.mutate(&config)
			if _, err := NewFrontend(config, test.budget); err == nil {
				t.Fatal("want refusal")
			}
		})
	}
	plan, err := NewFrontend(base, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var workspace Workspace
	if _, _, err := plan.Process(t.Context(), [][]float32{{1, 2, 3, 4}}, 7, &workspace, ProcessOptions{}); err == nil {
		t.Fatal("want sample-rate refusal")
	}
	if _, _, err := plan.Process(t.Context(), [][]float32{{1, float32(math.NaN()), 3, 4}}, 8, &workspace, ProcessOptions{}); err == nil {
		t.Fatal("want non-finite refusal")
	}
	cancelled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, _, err := plan.Process(cancelled, [][]float32{{1, 2, 3, 4}}, 8, &workspace, ProcessOptions{}); err == nil {
		t.Fatal("want cancellation")
	}
	other, err := NewFrontend(base, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := plan.Process(t.Context(), [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}, 8, &workspace, ProcessOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := other.Process(t.Context(), [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}, 8, &workspace, ProcessOptions{}); err == nil {
		t.Fatal("want workspace-owner refusal")
	}
}

func TestInverseRefusesUncoveredSamples(t *testing.T) {
	config := testFrontendConfig()
	config.PadLeft = 0
	config.PadRight = 0
	config.Padding = "zero"
	config.WindowOffset = 0
	config.FrameSpan = 4
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var w Workspace
	if values, err := plan.Reconstruct(t.Context(), [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}, 8, &w); err == nil || values != nil {
		t.Fatalf("uncovered inverse=%v error=%v", values, err)
	}
}

func TestWorkspaceBudgetRefusesBeforeAllocation(t *testing.T) {
	plan, err := NewFrontend(testFrontendConfig(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	plan.memoryBytes = plan.tableBytes
	var w Workspace
	if values, _, err := plan.Process(t.Context(), [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}, 8, &w, ProcessOptions{}); err == nil || values != nil {
		t.Fatal("want workspace budget refusal")
	}
	if cap(w.features) != 0 || cap(w.real) != 0 || w.owner != nil {
		t.Fatal("allocated workspace before budget approval")
	}
}

func TestOddFourierLengthAndEmptyChunkBoundaries(t *testing.T) {
	config := testFrontendConfig()
	config.FFTLength = 5
	config.FrameSpan = 5
	config.Geometry.WindowSamples = 5
	config.WindowOffset = 0
	config.PadLeft = 2
	config.PadRight = 2
	config.Window = "rectangular"
	plan, err := NewFrontend(config, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var w Workspace
	samples := []float32{.25, -.5, .75, -1, .125, -.25, .5}
	got, err := plan.Reconstruct(t.Context(), [][]float32{nil, samples[:2], nil, samples[2:], nil}, 8, &w)
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range samples {
		if got[index] != value {
			t.Fatalf("odd inverse[%d]=%g want %g", index, got[index], value)
		}
	}
}
