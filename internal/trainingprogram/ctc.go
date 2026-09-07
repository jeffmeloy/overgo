package trainingprogram

import (
	"context"
	"errors"
	"math"

	"overgo/internal/checked"
)

// CTCLossWorkspaceSize returns the float64 scratch elements needed for one
// sequence: forward states, two backward rows, row normalization and class
// posteriors. The caller admits and owns this storage; loss evaluation allocates
// none. Logits and an optional gradient each occupy frames*vocabulary float32s.
func CTCLossWorkspaceSize(frames, vocabulary, targets int) (int, error) {
	if frames <= 0 || vocabulary <= 0 || targets < 0 {
		return 0, errors.New("training program: invalid CTC geometry")
	}
	states, ok := checked.MulInt(targets, 2)
	if !ok {
		return 0, errors.New("training program: CTC state count overflows")
	}
	states, ok = checked.AddInt(states, 1)
	if !ok {
		return 0, errors.New("training program: CTC state count overflows")
	}
	rows, ok := checked.AddInt(frames, 2)
	if !ok {
		return 0, errors.New("training program: CTC row count overflows")
	}
	statesStorage, ok := checked.MulInt(rows, states)
	if !ok {
		return 0, errors.New("training program: CTC state storage overflows")
	}
	normalization, ok := checked.MulInt(frames, 2)
	if !ok {
		return 0, errors.New("training program: CTC normalization storage overflows")
	}
	count, ok := checked.AddInt(statesStorage, normalization, vocabulary)
	// A float64 occupies eight bytes; even a valid element count must fit
	// the addressable byte extent before the caller allocates its slice.
	if !ok || count > math.MaxInt/8 {
		return 0, errors.New("training program: CTC workspace overflows")
	}
	return count, nil
}

// CTCLossF32 computes one sequence's negative log alignment probability and,
// when gradient is non-nil, its raw-logits gradient. Logits are frame-major.
// The reduction is a sequence sum, not a target-length or batch average; a
// caller applies its declared averaging to both loss and gradient.
//
// Blanks separate repeated targets. Targets cannot contain blank. An empty
// target accepts only the all-blank path; zero frames and impossible alignments
// are refused, never converted to zero loss. Gradient must match logits without
// overlap. Scratch must meet CTCLossWorkspaceSize; no hidden allocation can
// exceed the caller's admitted storage. On any error, discard output and scratch.
//
// The forward/backward recurrence is evaluated in log space. Backward states
// exclude the current emission, so posterior accumulation needs only two rows
// rather than a second frames-by-states matrix.
func CTCLossF32(ctx context.Context, gradient, logits []float32, targets []int, frames, vocabulary, blank int, workspace []float64) (float64, error) {
	count, ok := checked.MulInt(frames, vocabulary)
	if ctx == nil || frames <= 0 || vocabulary <= 0 || !ok || count != len(logits) ||
		blank < 0 || blank >= vocabulary || gradient != nil && len(gradient) != count || checked.SlicesOverlap(gradient, logits) {
		return 0, errors.New("training program: invalid CTC context, shape, blank or gradient")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	minimum := len(targets)
	if minimum > frames {
		return 0, errors.New("training program: impossible CTC alignment")
	}
	for index, target := range targets {
		if target < 0 || target >= vocabulary || target == blank {
			return 0, errors.New("training program: invalid CTC target")
		}
		if index > 0 && target == targets[index-1] {
			if minimum == frames {
				return 0, errors.New("training program: impossible CTC repeated-target alignment")
			}
			minimum++
		}
	}
	needed, err := CTCLossWorkspaceSize(frames, vocabulary, len(targets))
	if err != nil {
		return 0, err
	}
	if len(workspace) < needed {
		return 0, errors.New("training program: CTC workspace is below the admitted geometry")
	}
	states := 2*len(targets) + 1
	alpha := workspace[:frames*states]
	maxima := workspace[len(alpha) : len(alpha)+frames]
	normalizers := workspace[len(alpha)+frames : len(alpha)+2*frames]
	beta := workspace[len(alpha)+2*frames : len(alpha)+2*frames+states]
	next := workspace[len(alpha)+2*frames+states : len(alpha)+2*frames+2*states]
	posterior := workspace[len(alpha)+2*frames+2*states : needed]
	for frame := range frames {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		row := logits[frame*vocabulary : (frame+1)*vocabulary]
		maximum := float64(row[0])
		for _, value := range row {
			if !checked.Finite32(value) {
				return 0, errors.New("training program: non-finite CTC logits")
			}
			maximum = max(maximum, float64(value))
		}
		var sum float64
		for _, value := range row {
			sum += math.Exp(float64(value) - maximum)
		}
		maxima[frame], normalizers[frame] = maximum, math.Log(sum)
	}
	label := func(state int) int {
		if state%2 == 0 {
			return blank
		}
		return targets[state/2]
	}
	logProbability := func(frame, state int) float64 {
		return (float64(logits[frame*vocabulary+label(state)]) - maxima[frame]) - normalizers[frame]
	}
	negativeInfinity := math.Inf(-1)
	for state := range states {
		alpha[state] = negativeInfinity
	}
	alpha[0] = logProbability(0, 0)
	if len(targets) != 0 {
		alpha[1] = logProbability(0, 1)
	}
	for frame := 1; frame < frames; frame++ {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		previous := alpha[(frame-1)*states : frame*states]
		for state := range states {
			mass := previous[state]
			if state > 0 {
				mass = ctcLogAdd(mass, previous[state-1])
			}
			if state > 1 && label(state) != label(state-2) {
				mass = ctcLogAdd(mass, previous[state-2])
			}
			alpha[frame*states+state] = mass + logProbability(frame, state)
		}
	}
	likelihood := alpha[len(alpha)-1]
	if len(targets) != 0 {
		likelihood = ctcLogAdd(likelihood, alpha[len(alpha)-2])
	}
	if !finite(likelihood) {
		return 0, errors.New("training program: non-finite CTC likelihood")
	}
	if gradient == nil {
		return -likelihood, nil
	}
	for state := range states {
		beta[state] = negativeInfinity
	}
	beta[states-1] = 0
	if len(targets) != 0 {
		beta[states-2] = 0
	}
	for frame := frames - 1; frame >= 0; frame-- {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		clear(posterior)
		for state := range states {
			posterior[label(state)] += math.Exp(alpha[frame*states+state] + beta[state] - likelihood)
		}
		for token := range vocabulary {
			index := frame*vocabulary + token
			probability := math.Exp((float64(logits[index]) - maxima[frame]) - normalizers[frame])
			gradient[index] = float32(probability - posterior[token])
		}
		if frame == 0 {
			break
		}
		for state := range states {
			mass := beta[state] + logProbability(frame, state)
			if state+1 < states {
				mass = ctcLogAdd(mass, beta[state+1]+logProbability(frame, state+1))
			}
			if state+2 < states && label(state) != label(state+2) {
				mass = ctcLogAdd(mass, beta[state+2]+logProbability(frame, state+2))
			}
			next[state] = mass
		}
		beta, next = next, beta
	}
	return -likelihood, nil
}

func ctcLogAdd(left, right float64) float64 {
	if left < right {
		left, right = right, left
	}
	if math.IsInf(left, -1) {
		return left
	}
	return left + softplus(right-left)
}
