package audiodsp

import (
	"context"
	"errors"

	"overgo/internal/checked"
	"overgo/internal/scratch"
)

// GroupedFeatureConfig declares hop-count selection and temporal composition.
// Floor(samples/hop) is rounded up to StackFrames; the waveform is zero-padded
// to (selectedFrames-1)*hop+FinalFrameSamples when necessary. Selection precedes
// global log scaling. DeltaRadius zero omits deltas; otherwise centered edge-
// replicated deltas are appended before consecutive frames are stacked.
type GroupedFeatureConfig struct {
	StackFrames       int `json:"stack_frames"`
	DeltaRadius       int `json:"delta_radius"`
	FinalFrameSamples int `json:"final_frame_samples"`
}

// GroupedFrames derives the encoder input-frame bound from a scalar sample
// count using the same selection geometry as ProcessGrouped. It allocates no
// waveform or feature storage and does not establish signal admission.
func (p *Frontend) GroupedFrames(samples uint64, config GroupedFeatureConfig) (int, error) {
	if p == nil || p.hop <= 0 || config.StackFrames <= 0 || config.DeltaRadius < 0 || config.FinalFrameSamples < 0 {
		return 0, errors.New("grouped features: invalid frame geometry")
	}
	hops := samples / uint64(p.hop)
	if hops == 0 {
		return 0, errors.New("grouped features: no selected frames")
	}
	groups := hops / uint64(config.StackFrames)
	if hops%uint64(config.StackFrames) != 0 {
		groups++
	}
	frames, ok := checked.Int(groups)
	if !ok {
		return 0, errors.New("grouped features: grouped frame extent overflows")
	}
	return frames, nil
}

// ProcessGrouped composes the existing frontend and optional delta operation.
// It returns borrowed frame-major data, grouped frame count and feature width.
// Input must already have the declared sample rate; resampling is a preceding
// operation. Padding and output storage share the frontend's numeric byte budget.
// It processes complete offline input, not a live streaming session.
func (p *Frontend) ProcessGrouped(ctx context.Context, input []float32, sampleRate int, w *Workspace, config GroupedFeatureConfig) ([]float32, int, int, error) {
	if p == nil || ctx == nil || w == nil || w.owner != nil && w.owner != p ||
		sampleRate <= 0 || sampleRate != p.config.SampleRate || p.hop <= 0 || p.bands <= 0 || config.StackFrames <= 0 || config.DeltaRadius < 0 || config.FinalFrameSamples < 0 {
		return nil, 0, 0, errors.New("grouped features: invalid execution or declaration")
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, 0, err
	}
	for _, storage := range [][]float32{w.sequence, w.padding, w.features, w.waveform, w.joined, w.resampled} {
		if checked.SlicesOverlap(input, storage[:cap(storage)]) {
			return nil, 0, 0, errors.New("grouped features: input aliases workspace")
		}
	}
	groups, err := p.GroupedFrames(uint64(len(input)), config)
	if err != nil {
		return nil, 0, 0, err
	}
	frames, ok := checked.MulInt(groups, config.StackFrames)
	if !ok {
		return nil, 0, 0, errors.New("grouped features: frame extent overflows")
	}
	minimum, ok := checked.MulInt(frames-1, p.hop)
	if !ok {
		return nil, 0, 0, errors.New("grouped features: waveform extent overflows")
	}
	minimum, ok = checked.AddInt(minimum, config.FinalFrameSamples)
	if !ok {
		return nil, 0, 0, errors.New("grouped features: waveform extent overflows")
	}
	padding := max(0, minimum-len(input))
	count, ok := checked.MulInt(frames, p.bands)
	expansion := 1
	if config.DeltaRadius > 0 {
		expansion++
	}
	width, widthOK := checked.ProductInt(p.bands, expansion, config.StackFrames)
	outputCount, outputOK := checked.MulInt(count, expansion)
	if !ok || !widthOK || !outputOK {
		return nil, 0, 0, errors.New("grouped features: feature extent overflows")
	}
	sequenceCount := 0
	if config.DeltaRadius > 0 {
		sequenceCount = outputCount
	}
	if err := p.reserveWorkspace(w, count, max(minimum, len(input)), false, sequenceCount, padding); err != nil {
		return nil, 0, 0, err
	}
	w.sequence = scratch.Resize(w.sequence, sequenceCount)
	w.padding = scratch.Resize(w.padding, padding)
	clear(w.padding)
	w.groupedChunks[0], w.groupedChunks[1] = input, w.padding
	features, selected, err := p.Process(ctx, w.groupedChunks[:], sampleRate, w, ProcessOptions{FrameLimit: frames})
	// Do not retain the caller's waveform after execution.
	w.groupedChunks[0], w.groupedChunks[1] = nil, nil
	if err != nil {
		return nil, 0, 0, err
	}
	if selected != frames {
		return nil, 0, 0, errors.New("grouped features: selected frame count changed")
	}
	if config.DeltaRadius > 0 {
		if err := AppendFrameDeltas(ctx, w.sequence, features, p.bands, config.DeltaRadius); err != nil {
			return nil, 0, 0, err
		}
		features = w.sequence
	}
	return features, groups, width, nil
}
