package projector

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/audiodsp"
	"overgo/internal/recipecontract"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func (r *Gemma4TowerRunner) EncodeAudio(
	ctx context.Context,
	samples []float32,
	sampleRate int,
	profile AudioProjectionProfile,
) (Gemma4AudioTowerOutput, error) {
	if r == nil || r.file == nil {
		return Gemma4AudioTowerOutput{}, errRunnerClosed
	}
	features, frames, err := r.audioPlan.preprocess(ctx, samples, sampleRate)
	if err != nil {
		return Gemma4AudioTowerOutput{}, err
	}
	return r.EncodeAudioFeatures(ctx, features, frames, profile)
}

func (r *Gemma4TowerRunner) EncodeAudioTrace(
	ctx context.Context,
	samples []float32,
	sampleRate int,
	profile AudioProjectionProfile,
) (Gemma4AudioTowerOutput, Gemma4AudioTowerTrace, error) {
	if r == nil || r.file == nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, errRunnerClosed
	}
	features, frames, err := r.audioPlan.preprocess(ctx, samples, sampleRate)
	if err != nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, err
	}
	output, trace, err := r.EncodeAudioFeaturesTrace(ctx, features, frames, profile)
	if err != nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, err
	}
	trace.Stages["frontend"] = reference.Value{
		Shape: tensor.MustShape(uint64(r.spec.Audio.MelBins), uint64(frames)),
		Data:  features,
	}
	return output, trace, nil
}

type audioFrontendPlan struct {
	frontend *audiodsp.Frontend
	err      error
}

// This adapter only maps the existing artifact declaration to the common
// frontend. The semicausal admission span retains its extra lookahead sample.
func newAudioFrontendPlan(spec Gemma4AudioTowerSpec) *audioFrontendPlan {
	config := audiodsp.FrontendConfig{
		SampleRate: spec.SampleRate,
		Geometry:   recipecontract.AudioFrameGeometry{WindowSamples: uint64(spec.FrameLength), HopSamples: uint64(spec.HopLength), FeatureBins: uint32(spec.MelBins)},
		FFTLength:  spec.FFTLength, FrameSpan: spec.FrameLength + 1, PadLeft: spec.FrameLength / 2,
		Padding: "zero", Window: "periodic-hann",
		Mel: audiodsp.MelConfig{Scale: "htk", MinFrequency: float64(spec.MinFrequency), MaxFrequency: float64(spec.MaxFrequency)},
		Log: audiodsp.LogConfig{Base: "natural", GuardMode: "add", Guard: float64(spec.MelFloor), Scale: 1},
	}
	// Existing projector admission supplies geometry, not a separate frontend
	// memory limit. Preserve that contract while checking native addressability.
	frontend, err := audiodsp.NewFrontend(config, uint64(math.MaxInt))
	return &audioFrontendPlan{frontend: frontend, err: err}
}

func (p *audioFrontendPlan) preprocess(ctx context.Context, samples []float32, sampleRate int) ([]float32, int, error) {
	if p == nil {
		return nil, 0, errors.New("projector: audio frontend plan is unavailable")
	}
	if p.err != nil {
		return nil, 0, fmt.Errorf("projector: audio frontend: %w", p.err)
	}
	// A workspace per invocation keeps the immutable plan safe for concurrent
	// requests. The returned feature storage is owned by this invocation.
	var workspace audiodsp.Workspace
	return p.frontend.Process(ctx, [][]float32{samples}, sampleRate, &workspace, audiodsp.ProcessOptions{})
}
