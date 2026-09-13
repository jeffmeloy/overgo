package latentvideo

import (
	"overgo/internal/media"
	"overgo/internal/tensor"
	"slices"
	"testing"
)

type layoutFixtureBackend struct{}

// ProjectBranchContexts supplies the fixture's two scalar branch coefficients.
func (layoutFixtureBackend) ProjectBranchContexts(conditional, unconditional []float32) (any, any, error) {
	return conditional[0], unconditional[0], nil
}

// ForwardHead consumes borrowed inputs synchronously and returns owned output.
func (layoutFixtureBackend) ForwardHead(patch, block, head []float32, branch any) ([]float32, error) {
	out := make([]float32, len(patch))
	for i, value := range patch {
		out[i] = value*branch.(float32) + head[0]
	}
	return out, nil
}

func layoutFixture(frames, height, width int) (*DenoiserProgram, DenoiseRequest) {
	// Two channels and a nonsquare grid exercise both layout permutations.
	dim := tensor.PairedExtent
	config := DenoiserConfig{Dim: dim, FreqDim: dim, InDim: dim, OutDim: dim, PatchSize: [3]int{1, 1, 2}, Policy: DenoiserPolicy{NumTrainTimesteps: 8, SinusoidalPeriod: 8, VAEStride: [3]int{1, 1, 1}}}
	geometry := LatentGeometry{Channels: dim, LatentFrames: frames, LatentHeight: height, LatentWidth: width, Grid: [3]int{frames, height, width / config.PatchSize[2]}, Seq: frames * height * width / config.PatchSize[2]}
	modulation := media.PairedShiftScaleGateWidth(dim)
	weights := TimestepConditioningWeights{Dim: dim, FreqDim: dim, Period: config.Policy.SinusoidalPeriod, Embed0W: make([]float32, dim*dim), Embed0B: make([]float32, dim), Embed2W: make([]float32, dim*dim), Embed2B: make([]float32, dim), ProjectW: make([]float32, modulation*dim), ProjectB: make([]float32, modulation)}
	sample := make([]float32, geometry.Elements())
	for i := range sample {
		sample[i] = float32(i) / float32(len(sample))
	}
	return &DenoiserProgram{Config: config, Geometry: geometry, timestepWeights: weights}, DenoiseRequest{Steps: 4, Shift: 1, GuideScale: 1.5, CondContext: []float32{0.1}, UncondContext: []float32{0.2}, InitialSample: sample}
}

func TestWanLayoutBufferOwnership(t *testing.T) {
	for _, shape := range [][3]int{{1, 2, 4}, {2, 2, 4}, {3, 4, 2}} {
		p, request := layoutFixture(shape[0], shape[1], shape[2])
		input := slices.Clone(request.InitialSample)
		first, err := p.PatchifyLatent(input)
		if err != nil {
			t.Fatal(err)
		}
		second, err := p.PatchifyLatent(input)
		if err != nil {
			t.Fatal(err)
		}
		first[0]++
		if second[0] == first[0] || !slices.Equal(input, request.InitialSample) {
			t.Fatal("public patch output aliases another output or its input")
		}
		decoded, err := p.UnpatchifyLatent(second)
		if err != nil {
			t.Fatal(err)
		}
		another, err := p.UnpatchifyLatent(second)
		if err != nil {
			t.Fatal(err)
		}
		decoded[0]++
		if decoded[0] == another[0] {
			t.Fatal("public unpatch output aliases another output")
		}
		if _, err := p.PatchifyLatent(input[:len(input)-1]); err == nil {
			t.Fatal("accepted short latent")
		}
		if _, err := p.UnpatchifyLatent(second[:len(second)-1]); err == nil {
			t.Fatal("accepted short head")
		}
		plain, err := p.DenoiseWithBackend(layoutFixtureBackend{}, request)
		if err != nil {
			t.Fatal(err)
		}
		request.TraceSteps = true
		traced, err := p.DenoiseWithBackend(layoutFixtureBackend{}, request)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(plain.Latent, traced.Latent) || len(traced.Steps) != request.Steps {
			t.Fatal("tracing changed the trajectory")
		}
		saved := make([][]float32, len(traced.Steps))
		for i, step := range traced.Steps {
			saved[i] = slices.Clone(step.CondOutput)
		}
		for i := range traced.Steps {
			traced.Steps[i].CondOutput[0]++
			for j := i + 1; j < len(traced.Steps); j++ {
				if !slices.Equal(traced.Steps[j].CondOutput, saved[j]) {
					t.Fatal("later trace aliases an earlier branch output")
				}
			}
		}
		if !slices.Equal(input, request.InitialSample) {
			t.Fatal("denoising overwrote borrowed initial sample")
		}
	}
}
