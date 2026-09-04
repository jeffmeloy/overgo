package audiodsp

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func BenchmarkFrontend(b *testing.B) {
	encoded, err := os.ReadFile("testdata/power_logmel_frontend.json")
	if err != nil {
		b.Fatal(err)
	}
	var config FrontendConfig
	if err := json.Unmarshal(encoded, &config); err != nil {
		b.Fatal(err)
	}
	// Fixed synthetic input, not an ASR quality or real-time serving claim.
	samples := make([]float32, config.SampleRate)
	for index := range samples {
		samples[index] = float32(index%config.FFTLength)/float32(config.FFTLength) - 0.5
	}
	b.Run("cold-plan", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := NewFrontend(config, uint64(math.MaxInt)); err != nil {
				b.Fatal(err)
			}
		}
	})
	for _, chunkSize := range []int{len(samples), int(config.Geometry.HopSamples), 1} {
		chunks := make([][]float32, 0, (len(samples)+chunkSize-1)/chunkSize)
		for start := 0; start < len(samples); start += chunkSize {
			chunks = append(chunks, samples[start:min(start+chunkSize, len(samples))])
		}
		name := "whole"
		if chunkSize == 1 {
			name = "single-sample-chunks"
		} else if chunkSize != len(samples) {
			name = "hop-chunks"
		}
		b.Run(name, func(b *testing.B) {
			plan, err := NewFrontend(config, uint64(math.MaxInt))
			if err != nil {
				b.Fatal(err)
			}
			var workspace Workspace
			ctx := b.Context()
			if _, _, err := plan.Process(ctx, chunks, config.SampleRate, &workspace, ProcessOptions{}); err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(len(samples)) * float32Bytes)
			b.ReportAllocs()
			for b.Loop() {
				if _, _, err := plan.Process(ctx, chunks, config.SampleRate, &workspace, ProcessOptions{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestFrontendSteadyStateAllocations(t *testing.T) {
	samples := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	for _, resample := range []bool{false, true} {
		name := "native-rate"
		config, sampleRate := testFrontendConfig(), 8
		if resample {
			name, sampleRate = "resampled", 4
			config.ResampleTaps = []float64{0.5, 1, 0.5}
		}
		t.Run(name, func(t *testing.T) {
			plan, err := NewFrontend(config, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			chunks := [][]float32{nil, samples[:3], nil, samples[3:], nil}
			ctx := t.Context()
			var workspace Workspace
			for _, inverse := range []bool{false, true} {
				run := func() {
					var err error
					if inverse {
						_, err = plan.Reconstruct(ctx, chunks, sampleRate, &workspace)
					} else {
						_, _, err = plan.Process(ctx, chunks, sampleRate, &workspace, ProcessOptions{})
					}
					if err != nil {
						t.Fatal(err)
					}
				}
				run()
				if allocations := testing.AllocsPerRun(100, run); allocations != 0 {
					t.Errorf("inverse=%v steady-state allocations=%g, want zero", inverse, allocations)
				}
			}
		})
	}
}
