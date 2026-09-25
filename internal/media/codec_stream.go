package media

import (
	"errors"
	"slices"
	"sync"

	"overgo/internal/checked"
)

// CodecChunkLineage records contiguous input/output intervals within one stream.
// These operational offsets are not independent statistical units or source IDs;
// the family/request owner retains the immutable source authority.
type CodecChunkLineage struct {
	Ordinal, InputStart, InputFrames, OutputStart, OutputFrames int
}

// CodecChunkStream owns per-operation temporal state for one immutable program.
// Its mutex protects only this stream's calls; it reserves no device or worktree.
// A failed operation poisons continuation until Reset discards temporal state.
type CodecChunkStream[Bindings, State any] struct {
	mu      sync.Mutex
	program CodecProgram[Bindings]
	states  []State
	next    CodecChunkLineage
	failed  bool
}

// NewCodecChunkStream starts a cold stream using the existing program owner.
// Bindings remain immutable and family-owned; the operation slice is copied.
func NewCodecChunkStream[Bindings, State any](program CodecProgram[Bindings]) (*CodecChunkStream[Bindings, State], error) {
	if err := program.Validate("codec stream"); err != nil {
		return nil, err
	}
	program.Operations = slices.Clone(program.Operations)
	return &CodecChunkStream[Bindings, State]{program: program, states: make([]State, len(program.Operations))}, nil
}

// Reset starts a new cold stream. Allocation/workspace release stays with its
// family owner; cleared temporal state cannot inherit a previous clip's cache.
func (stream *CodecChunkStream[Bindings, State]) Reset() {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	clear(stream.states)
	stream.next = CodecChunkLineage{}
	stream.failed = false
}

// ExecuteCodecChunk checks order and intervals, then delegates numerical stage
// execution to ExecuteCodecProgram. A refusal before execution preserves state;
// any partial execution failure requires Reset before reuse.
func ExecuteCodecChunk[Bindings, Storage, State any](scope string, stream *CodecChunkStream[Bindings, State], ordinal, inputStart int, input CodecVolume[Storage], step CodecStep[Bindings, Storage, State]) (CodecVolume[Storage], CodecChunkLineage, error) {
	var lineage CodecChunkLineage
	if stream == nil {
		return input, lineage, errors.New("media: nil codec stream")
	}
	stream.mu.Lock()
	defer stream.mu.Unlock()
	if stream.failed || ordinal != stream.next.Ordinal || inputStart != stream.next.InputStart {
		return input, lineage, errors.New("media: codec chunk has stale, skipped, or failed predecessor")
	}
	inputEnd, inputOK := checked.AddInt(inputStart, input.Frames)
	nextOrdinal, ordinalOK := checked.AddInt(ordinal, 1)
	if !inputOK || !ordinalOK || input.Frames <= 0 {
		return input, lineage, errors.New("media: codec chunk input interval overflows")
	}
	output, err := ExecuteCodecProgram(scope, stream.program, stream.states, input, ordinal > 0, step)
	if err != nil {
		stream.failed = true
		return output, lineage, err
	}
	outputEnd, ok := checked.AddInt(stream.next.OutputStart, output.Frames)
	if !ok {
		stream.failed = true
		return output, lineage, errors.New("media: codec chunk output interval overflows")
	}
	lineage = CodecChunkLineage{Ordinal: ordinal, InputStart: inputStart, InputFrames: input.Frames, OutputStart: stream.next.OutputStart, OutputFrames: output.Frames}
	stream.next = CodecChunkLineage{Ordinal: nextOrdinal, InputStart: inputEnd, OutputStart: outputEnd}
	return output, lineage, nil
}
