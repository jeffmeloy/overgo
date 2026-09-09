// Copyright 2026 ShugoAI LLC
// Licensed under the Apache License, Version 2.0.
// Derived from audio.cpp 3497b7cc44753e2c141d8fe60ac42cec433e3281,
// src/community_models/mms_forced_aligner/ctc_alignment.cpp.
// Modified for neutral Go execution, caller-owned storage, cancellation and
// byte admission. No model geometry or text normalization is inherited.

package speechrecognition

import (
	"context"
	"errors"
	"math"
	"strconv"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/scratch"
)

// CTCAlignment contains the best path through [blank, target, blank, ...].
// States borrows the workspace until its next invocation. Score is the sum of
// selected float32 log emissions, not a calibrated confidence or CTC loss.
type CTCAlignment struct {
	States []int
	Score  float32
}

// CTCAlignmentWorkspace owns two score rows, packed two-bit backpointers and
// the resulting path. Its zero value is ready for sequential use. Memory
// admission covers retained numeric storage, not allocator or process peak.
type CTCAlignmentWorkspace struct {
	scores []float32
	moves  []byte
	path   []int
}

const ctcMovesPerByte = 4 // Three possible moves need two bits; eight bits hold four cells.

func (w *CTCAlignmentWorkspace) discardExcess(memoryBytes uint64) {
	bytes, ok := checked.Add64(uint64(cap(w.scores))*binaryschema.Uint32Bytes,
		uint64(cap(w.moves)), uint64(cap(w.path))*(strconv.IntSize/8))
	if !ok || bytes > memoryBytes {
		*w = CTCAlignmentWorkspace{}
	}
}

// Align finds the maximum-score CTC path for a nonempty target. Emissions are
// frame-major log probabilities: negative infinity is allowed, NaN, positive
// infinity and positive values are not. Rows are not renormalized. Blanks must
// separate repeated targets. Ties prefer stay, advance-one, then advance-two;
// the final target wins a tie with the final blank. This matches the pinned
// oracle. On error, discard prior borrowed results.
func (w *CTCAlignmentWorkspace) Align(ctx context.Context, emissions []float32, targets []int, frames, vocabulary, blank int, memoryBytes uint64) (CTCAlignment, error) {
	count, ok := checked.MulInt(frames, vocabulary)
	if w == nil || ctx == nil || frames <= 0 || vocabulary <= 0 || !ok || count != len(emissions) ||
		blank < 0 || blank >= vocabulary || len(targets) == 0 || len(targets) > frames || checked.SlicesOverlap(targets, w.path[:cap(w.path)]) {
		return CTCAlignment{}, errors.New("CTC alignment: invalid context, shape, blank or target storage")
	}
	if err := ctx.Err(); err != nil {
		return CTCAlignment{}, err
	}
	minimum := len(targets)
	for index, target := range targets {
		if target < 0 || target >= vocabulary || target == blank {
			return CTCAlignment{}, errors.New("CTC alignment: invalid target")
		}
		if index > 0 && target == targets[index-1] {
			if minimum == frames {
				return CTCAlignment{}, errors.New("CTC alignment: insufficient frames for repeated targets")
			}
			minimum++
		}
	}
	states, moves, bytes, err := alignmentStorage(frames, len(targets))
	if err != nil {
		return CTCAlignment{}, err
	}
	if bytes > memoryBytes {
		return CTCAlignment{}, errors.New("CTC alignment: numeric storage exceeds byte admission")
	}
	for frame := range frames {
		if err := ctx.Err(); err != nil {
			return CTCAlignment{}, err
		}
		for _, value := range emissions[frame*vocabulary : (frame+1)*vocabulary] {
			if math.IsNaN(float64(value)) || value > 0 {
				return CTCAlignment{}, errors.New("CTC alignment: invalid log emission")
			}
		}
	}
	// Mixed shape changes can exceed admission even when each shape alone
	// fits: account for the capacities that resize would retain, not lengths.
	retained, ok := checked.Add64(uint64(max(cap(w.scores), 2*states))*binaryschema.Uint32Bytes,
		uint64(max(cap(w.moves), moves)), uint64(max(cap(w.path), frames))*(strconv.IntSize/8))
	if !ok || retained > memoryBytes {
		*w = CTCAlignmentWorkspace{}
	}
	w.scores = scratch.Resize(w.scores, 2*states)
	w.moves = scratch.Resize(w.moves, moves)
	w.path = scratch.Resize(w.path, frames)
	previous, current := w.scores[:states], w.scores[states:]
	for state := range previous {
		previous[state] = float32(math.Inf(-1))
	}
	previous[0], previous[1] = emissions[blank], emissions[targets[0]]
	label := func(state int) int {
		if state%2 == 0 {
			return blank
		}
		return targets[state/2]
	}
	for frame := 1; frame < frames; frame++ {
		if err := ctx.Err(); err != nil {
			return CTCAlignment{}, err
		}
		for state := range states {
			best, move := previous[state], byte(0)
			if state > 0 && previous[state-1] > best {
				best, move = previous[state-1], 1
			}
			if state%2 == 1 && state >= 3 && targets[state/2] != targets[state/2-1] && previous[state-2] > best {
				best, move = previous[state-2], 2
			}
			current[state] = best + emissions[frame*vocabulary+label(state)]
			cell := frame*states + state
			shift := uint(cell%ctcMovesPerByte) * 2
			w.moves[cell/ctcMovesPerByte] = w.moves[cell/ctcMovesPerByte]&^(3<<shift) | move<<shift
		}
		previous, current = current, previous
	}
	state := states - 2
	if previous[states-1] > previous[state] {
		state = states - 1
	}
	score := previous[state]
	if !checked.Finite32(score) {
		return CTCAlignment{}, errors.New("CTC alignment: no finite complete path")
	}
	for frame := frames - 1; frame >= 0; frame-- {
		if err := ctx.Err(); err != nil {
			return CTCAlignment{}, err
		}
		w.path[frame] = state
		if frame > 0 {
			cell := frame*states + state
			state -= int(w.moves[cell/ctcMovesPerByte] >> (uint(cell%ctcMovesPerByte) * 2) & 3)
		}
	}
	return CTCAlignment{States: w.path, Score: score}, ctx.Err()
}

func alignmentStorage(frames, targets int) (states, moves int, bytes uint64, err error) {
	if frames <= 0 || targets <= 0 {
		return 0, 0, 0, errors.New("CTC alignment: positive workspace geometry required")
	}
	states, ok := checked.MulInt(targets, 2)
	if ok {
		states, ok = checked.AddInt(states, 1)
	}
	cells, cellsOK := checked.MulInt(frames, states)
	if !ok || !cellsOK || states > math.MaxInt/(2*binaryschema.Uint32Bytes) || frames > math.MaxInt/(strconv.IntSize/8) {
		return 0, 0, 0, errors.New("CTC alignment: workspace extent overflows")
	}
	moves = (cells-1)/ctcMovesPerByte + 1
	bytes, ok = checked.Add64(uint64(states)*(2*binaryschema.Uint32Bytes), uint64(frames)*(strconv.IntSize/8), uint64(moves))
	if !ok {
		return 0, 0, 0, errors.New("CTC alignment: workspace byte count overflows")
	}
	return states, moves, bytes, nil
}
