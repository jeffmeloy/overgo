package speechactivity

import (
	"context"
	"errors"
	"math"

	"overgo/internal/checked"
)

// BoundaryConfig declares probability smoothing and speech/silence hysteresis.
// Durations are frame counts. Start padding includes at least the smoothing
// window. All values are explicit recipe facts, not model-family defaults.
type BoundaryConfig struct {
	Smoothing  int     `json:"smoothing"`
	Threshold  float64 `json:"threshold"`
	PadStart   uint64  `json:"pad_start"`
	MinSpeech  uint64  `json:"min_speech"`
	MaxSpeech  uint64  `json:"max_speech"`
	MinSilence uint64  `json:"min_silence"`
}

type boundaryPhase uint8

const (
	quiet boundaryPhase = iota
	possibleSpeech
	speaking
	possibleSilence
)

// BoundaryState contains the complete finite streaming decision state. Frame,
// Start and End use one-based frame indices; zero Start/End means absent.
// Window is a ring whose oldest entry is at Next when full. The profile owning
// a persisted state must bind the exact BoundaryConfig before restoring it.
type BoundaryState struct {
	Frame   uint64        `json:"frame"`
	Window  []float64     `json:"window"`
	Next    int           `json:"next"`
	Sum     float64       `json:"sum"`
	Phase   boundaryPhase `json:"phase"`
	Speech  uint64        `json:"speech"`
	Silence uint64        `json:"silence"`
	Split   bool          `json:"split"`
	Start   uint64        `json:"start"`
	End     uint64        `json:"end"`
	Final   bool          `json:"final"`
}

// Boundary records the threshold decision and optional confirmed boundaries.
// Probabilities are unrounded. Frame indices are one-based; a start at S maps
// to sample (S-1)*hop. End uses the same mapping, matching live frame events.
type Boundary struct {
	Frame       uint64  `json:"frame"`
	Probability float64 `json:"probability"`
	Smoothed    float64 `json:"smoothed"`
	Speech      bool    `json:"speech"`
	Started     bool    `json:"started"`
	Ended       bool    `json:"ended"`
	Start       uint64  `json:"start"`
	End         uint64  `json:"end"`
}

// BoundaryPolicy is immutable streaming hysteresis over declared probabilities.
type BoundaryPolicy struct{ config BoundaryConfig }

// NewBoundaryPolicy validates geometry and the numeric smoothing-state budget.
func NewBoundaryPolicy(config BoundaryConfig, memoryBytes uint64) (*BoundaryPolicy, error) {
	if config.Smoothing <= 0 || !checked.UnitInterval64(config.Threshold) || config.MinSpeech == 0 || config.MaxSpeech < config.MinSpeech || config.MinSilence == 0 ||
		uint64(config.Smoothing) > memoryBytes/8 {
		return nil, errors.New("speech boundary: invalid policy or smoothing budget")
	}
	config.PadStart = max(config.PadStart, uint64(config.Smoothing))
	return &BoundaryPolicy{config: config}, nil
}

func (p *BoundaryPolicy) validate(state *BoundaryState) error {
	if p == nil || p.config.Smoothing <= 0 || state == nil || state.Final || state.Phase > possibleSilence ||
		len(state.Window) > p.config.Smoothing || state.Next < 0 || state.Next >= p.config.Smoothing ||
		len(state.Window) < p.config.Smoothing && state.Next != 0 || !checked.Finite64(state.Sum) ||
		state.Start > state.Frame || state.End > state.Frame || state.Speech > state.Frame || state.Silence > state.Frame {
		return errors.New("speech boundary: invalid restart state")
	}
	for _, value := range state.Window {
		if !checked.UnitInterval64(value) {
			return errors.New("speech boundary: invalid smoothing history")
		}
	}
	if p.config.Smoothing > 1 && uint64(len(state.Window)) != min(state.Frame, uint64(p.config.Smoothing)) ||
		p.config.Smoothing == 1 && (len(state.Window) != 0 || state.Sum != 0 || state.Next != 0) {
		return errors.New("speech boundary: incomplete smoothing history")
	}
	return nil
}

// Process advances caller-owned state synchronously. visit may observe each
// frame but must not mutate state or reenter Process. An error invalidates the
// in-flight state; resume from the last completed artifact, not partial work.
// A final flush closes an open interval at the last observed frame, without
// inventing probabilities or a padded frame. No recording prefix is retained.
func (p *BoundaryPolicy) Process(ctx context.Context, probabilities []float32, state *BoundaryState, final bool, visit func(Boundary) error) error {
	if ctx == nil {
		return errors.New("speech boundary: missing context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := p.validate(state); err != nil {
		return err
	}
	if uint64(len(probabilities)) > math.MaxUint64-state.Frame {
		return errors.New("speech boundary: frame extent overflows")
	}
	for _, value := range probabilities {
		if !checked.UnitInterval64(float64(value)) {
			return errors.New("speech boundary: probability outside unit interval")
		}
	}
	for _, value := range probabilities {
		if err := ctx.Err(); err != nil {
			return err
		}
		state.Frame++
		smoothed := p.smooth(float64(value), state)
		result := Boundary{Frame: state.Frame, Probability: float64(value), Smoothed: smoothed, Speech: smoothed >= p.config.Threshold}
		p.advance(state, &result)
		if visit != nil {
			if err := visit(result); err != nil {
				return err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if final {
		if state.Start != 0 && visit != nil {
			if err := visit(Boundary{Frame: state.Frame, Ended: true, Start: state.Start, End: state.Frame}); err != nil {
				return err
			}
		}
		state.Final = true
	}
	return ctx.Err()
}

func (p *BoundaryPolicy) smooth(value float64, state *BoundaryState) float64 {
	width := p.config.Smoothing
	if width == 1 {
		return value
	}
	// Match add-then-remove order across both execution and serialized restart.
	state.Sum += value
	if cap(state.Window) != width {
		next := make([]float64, len(state.Window), width)
		copy(next, state.Window)
		state.Window = next
	}
	if len(state.Window) < width {
		state.Window = append(state.Window, value)
	} else {
		state.Sum -= state.Window[state.Next]
		state.Window[state.Next] = value
		state.Next = (state.Next + 1) % width
	}
	return state.Sum / float64(len(state.Window))
}

func (p *BoundaryPolicy) advance(state *BoundaryState, result *Boundary) {
	c := p.config
	if state.Split {
		result.Started, result.Start = true, state.Frame
		state.Start, state.Split = state.Frame, false
	}
	switch state.Phase {
	case quiet:
		if result.Speech {
			state.Phase = possibleSpeech
			state.Speech++
		} else {
			state.Silence++
			state.Speech = 0
		}
	case possibleSpeech:
		if !result.Speech {
			state.Phase = quiet
			state.Silence = 1
			state.Speech = 0
			break
		}
		state.Speech++
		if state.Speech >= c.MinSpeech {
			state.Phase = speaking
			start := state.Frame - state.Speech + 1
			if start <= c.PadStart {
				start = 1
			} else {
				start -= c.PadStart
			}
			result.Started, result.Start = true, max(start, state.End+1)
			state.Start, state.Silence = result.Start, 0
		}
	case speaking, possibleSilence:
		state.Speech++
		if result.Speech {
			state.Phase, state.Silence = speaking, 0
			if state.Speech >= c.MaxSpeech {
				state.Split, state.Speech = true, 0
				p.end(state, result)
			}
		} else {
			previous := state.Phase
			state.Phase = possibleSilence
			state.Silence++
			if previous == possibleSilence && state.Silence >= c.MinSilence {
				state.Phase, state.Speech = quiet, 0
				p.end(state, result)
			}
		}
	}
}

func (p *BoundaryPolicy) end(state *BoundaryState, result *Boundary) {
	result.Ended, result.End, result.Start = true, state.Frame, state.Start
	state.Start, state.End = 0, state.Frame
}
