package audiodsp

import (
	"context"
	"errors"
	"math"

	"overgo/internal/checked"
)

// StreamState is a completed framing boundary. Tail retains the next frame's
// raw samples and, when necessary, its preemphasis predecessor. Persisted state
// must be bound to the exact frontend declaration by its owning artifact profile.
type StreamState struct {
	Samples uint64    `json:"samples"`
	Frames  uint64    `json:"frames"`
	Tail    []float32 `json:"tail"`
	Final   bool      `json:"final"`
}

// StreamFrontend applies a frame-local frontend to live mono input.
// Resampling and recording-wide normalization require separate stateful owners;
// they are refused here. Declared zero padding is applied once at each end;
// right padding is withheld until finalization, never inserted between chunks.
type StreamFrontend struct {
	frontend *Frontend
	history  int
	tailSize int
}

// StreamWorkspace owns borrowed output and a geometry-bounded raw tail. Its
// zero value is usable. Each concurrent stream requires its own workspace.
type StreamWorkspace struct {
	frontend Workspace
	tail     []float32
}

// NewStreamFrontend validates live semantics and budgets retained overlap in
// addition to the existing frontend's tables and execution workspace.
func NewStreamFrontend(config FrontendConfig, memoryBytes uint64) (*StreamFrontend, error) {
	if (config.PadLeft != 0 || config.PadRight != 0) && config.Padding != "zero" || len(config.ResampleTaps) != 0 || config.Log.DynamicRange != nil ||
		config.Normalize != nil && config.Normalize.Mode != "fixed" {
		return nil, errors.New("audio stream frontend: requires zero-padded frame-local transforms")
	}
	p, err := NewFrontend(config, memoryBytes)
	if err != nil {
		return nil, err
	}
	if p.hop > p.config.FrameSpan {
		return nil, errors.New("audio stream frontend: hop exceeds frame span")
	}
	if config.PadLeft >= config.FrameSpan || config.PadRight >= config.FrameSpan ||
		config.MaskIncompleteHop && config.FrameSpan-config.PadLeft < p.hop {
		return nil, errors.New("audio stream frontend: padding would publish incomplete-hop frames before finalization")
	}
	stream := &StreamFrontend{frontend: p, tailSize: config.FrameSpan - 1}
	if config.WaveformPreemphasis != nil && config.WindowOffset == 0 {
		stream.history = 1
		stream.tailSize++
	}
	if err := p.reserve(p.tableBytes, nil, []int{stream.tailSize}); err != nil {
		return nil, err
	}
	p.memoryBytes -= uint64(stream.tailSize) * float32Bytes
	return stream, nil
}

func (p *StreamFrontend) frameCount(samples uint64, final bool) (uint64, error) {
	if samples == 0 {
		return 0, nil
	}
	f := p.frontend
	padding := uint64(f.config.PadLeft)
	if final {
		padding += uint64(f.config.PadRight)
	}
	if samples > math.MaxUint64-padding {
		return 0, errors.New("audio stream frontend: padded sample extent overflows")
	}
	if samples+padding < uint64(f.config.FrameSpan) {
		return 0, nil
	}
	return 1 + (samples+padding-uint64(f.config.FrameSpan))/uint64(f.hop), nil
}

// tailStart uses the non-final frame count, whose next origin cannot exceed
// the sample extent. Keep a raw predecessor rather than rounding a filtered tail.
func (p *StreamFrontend) tailStart(frames uint64) uint64 {
	step := frames * uint64(p.frontend.hop)
	prefix := uint64(p.frontend.config.PadLeft) + uint64(p.history)
	if step <= prefix {
		return 0
	}
	return step - prefix
}

func (p *StreamFrontend) validateState(state StreamState) error {
	frames, err := p.frameCount(state.Samples, false)
	if err != nil {
		return err
	}
	if state.Final || state.Frames != frames || state.Samples-p.tailStart(frames) != uint64(len(state.Tail)) || len(state.Tail) > p.tailSize {
		return errors.New("audio stream frontend: invalid or finalized restart boundary")
	}
	for _, value := range state.Tail {
		if !checked.Finite32(value) {
			return errors.New("audio stream frontend: non-finite restart samples")
		}
	}
	return nil
}

// Process consumes only new samples, returning complete-frame features and a
// restart boundary. Results borrow w until its next call; externally owned prior
// state and input remain unchanged. Empty input is legal only for finalization.
// Errors return no usable result; discard mutated scratch, not the last durable
// checkpoint. The caller serializes use and binds state to its model/profile.
func (p *StreamFrontend) Process(ctx context.Context, samples []float32, sampleRate int, previous StreamState, final bool, w *StreamWorkspace) ([]float32, int, StreamState, error) {
	if p == nil || p.frontend == nil || ctx == nil || w == nil || sampleRate != p.frontend.config.SampleRate || len(samples) == 0 && !final {
		return nil, 0, StreamState{}, errors.New("audio stream frontend: incomplete invocation")
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, StreamState{}, err
	}
	if err := p.validateState(previous); err != nil {
		return nil, 0, StreamState{}, err
	}
	if w.frontend.owner != nil && w.frontend.owner != p.frontend || checked.SlicesOverlap(samples, w.tail[:cap(w.tail)]) {
		return nil, 0, StreamState{}, errors.New("audio stream frontend: wrong workspace or input alias")
	}
	count, ok := checked.AddInt(len(previous.Tail), len(samples))
	if !ok || uint64(len(samples)) > math.MaxUint64-previous.Samples {
		return nil, 0, StreamState{}, errors.New("audio stream frontend: sample extent overflows")
	}
	for _, value := range samples {
		if !checked.Finite32(value) {
			return nil, 0, StreamState{}, errors.New("audio stream frontend: non-finite samples")
		}
	}
	f := p.frontend
	for _, input := range [][]float32{previous.Tail, samples} {
		for _, scratch := range [][]float32{w.frontend.features, w.frontend.waveform, w.frontend.joined, w.frontend.resampled, w.frontend.sequence} {
			if checked.SlicesOverlap(input, scratch[:cap(scratch)]) {
				return nil, 0, StreamState{}, errors.New("audio stream frontend: input aliases mutable frontend workspace")
			}
		}
	}
	total := previous.Samples + uint64(len(samples))
	complete, err := p.frameCount(total, final)
	if err != nil {
		return nil, 0, StreamState{}, err
	}
	frames, ok := checked.Int(complete - previous.Frames)
	if !ok {
		return nil, 0, StreamState{}, errors.New("audio stream frontend: frame extent overflows")
	}
	// Both source coordinates and Fourier positions must remain addressable.
	if _, ok := checked.AddInt(count, f.config.PadRight, f.config.FrameSpan); !ok {
		return nil, 0, StreamState{}, errors.New("audio stream frontend: padded frame position overflows")
	}
	origin := -f.config.PadLeft
	step := previous.Frames * uint64(f.hop)
	if step >= uint64(f.config.PadLeft) {
		origin = int(step - uint64(f.config.PadLeft) - p.tailStart(previous.Frames))
	} else {
		origin += int(step)
	}
	var features []float32
	if frames != 0 {
		source := chunkSource{chunks: [][]float32{previous.Tail, samples}, count: count, frameOrigin: origin}
		valid := frames
		if f.config.MaskIncompleteHop {
			valid = int(min(uint64(frames), total/uint64(f.hop)-previous.Frames))
		}
		features, _, err = f.processFrames(ctx, source, frames, valid, &w.frontend, ProcessOptions{})
		if err != nil {
			return nil, 0, StreamState{}, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, StreamState{}, err
	}
	keep := 0
	if !final {
		keep = int(total - p.tailStart(complete))
	}
	if w.tail == nil {
		w.tail = make([]float32, 0, p.tailSize)
	}
	w.tail = w.tail[:keep]
	fromPrevious := max(0, keep-len(samples))
	copy(w.tail[:fromPrevious], previous.Tail[len(previous.Tail)-fromPrevious:])
	copy(w.tail[fromPrevious:], samples[max(0, len(samples)-keep):])
	w.frontend.owner = f
	state := StreamState{Samples: total, Frames: complete, Tail: w.tail, Final: final}
	return features, frames, state, nil
}
