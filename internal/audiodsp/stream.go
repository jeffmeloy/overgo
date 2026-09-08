package audiodsp

import (
	"context"
	"errors"
	"math"

	"overgo/internal/checked"
)

// StreamState is a completed unpadded framing boundary. Tail starts at the next
// frame's sample offset, not the last input chunk boundary. Persisted state must
// be bound to the exact frontend declaration by its owning artifact profile.
type StreamState struct {
	Samples uint64    `json:"samples"`
	Frames  uint64    `json:"frames"`
	Tail    []float32 `json:"tail"`
	Final   bool      `json:"final"`
}

// StreamFrontend applies an unpadded, frame-local frontend to live mono input.
// Resampling and recording-wide normalization require separate stateful owners;
// they are refused here. An incomplete final frame is discarded without padding.
type StreamFrontend struct{ frontend *Frontend }

// StreamWorkspace owns borrowed output and a tail shorter than FrameSpan. Its
// zero value is usable. Each concurrent stream requires its own workspace.
type StreamWorkspace struct {
	frontend Workspace
	tail     []float32
}

// NewStreamFrontend validates live semantics and budgets retained overlap in
// addition to the existing frontend's tables and execution workspace.
func NewStreamFrontend(config FrontendConfig, memoryBytes uint64) (*StreamFrontend, error) {
	if config.PadLeft != 0 || config.PadRight != 0 || len(config.ResampleTaps) != 0 || config.Log.DynamicRange != nil ||
		config.Normalize != nil && config.Normalize.Mode != "fixed" {
		return nil, errors.New("audio stream frontend: requires unpadded frame-local transforms")
	}
	p, err := NewFrontend(config, memoryBytes)
	if err != nil {
		return nil, err
	}
	if p.hop > p.config.FrameSpan {
		return nil, errors.New("audio stream frontend: hop exceeds frame span")
	}
	if err := p.reserve(p.tableBytes, nil, []int{p.config.FrameSpan - 1}); err != nil {
		return nil, err
	}
	p.memoryBytes -= uint64(p.config.FrameSpan-1) * float32Bytes
	return &StreamFrontend{frontend: p}, nil
}

func (p *StreamFrontend) validateState(state StreamState) error {
	f := p.frontend
	frames := uint64(0)
	if state.Samples >= uint64(f.config.FrameSpan) {
		frames = 1 + (state.Samples-uint64(f.config.FrameSpan))/uint64(f.hop)
	}
	if state.Final || state.Frames != frames || state.Samples-frames*uint64(f.hop) != uint64(len(state.Tail)) || len(state.Tail) >= f.config.FrameSpan {
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
	var features []float32
	frames := 0
	if count >= f.config.FrameSpan {
		var err error
		features, frames, err = f.Process(ctx, [][]float32{previous.Tail, samples}, sampleRate, &w.frontend, ProcessOptions{})
		if err != nil {
			return nil, 0, StreamState{}, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, StreamState{}, err
	}
	keep := count - frames*f.hop
	if w.tail == nil {
		w.tail = make([]float32, 0, f.config.FrameSpan-1)
	}
	w.tail = w.tail[:keep]
	fromPrevious := max(0, keep-len(samples))
	copy(w.tail[:fromPrevious], previous.Tail[len(previous.Tail)-fromPrevious:])
	copy(w.tail[fromPrevious:], samples[max(0, len(samples)-keep):])
	w.frontend.owner = f
	state := StreamState{Samples: previous.Samples + uint64(len(samples)), Frames: previous.Frames + uint64(frames), Tail: w.tail, Final: final}
	return features, frames, state, nil
}
