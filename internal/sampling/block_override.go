package sampling

import (
	"errors"

	"overgo/internal/checked"
)

// BlockOverride is what a guider changes of a DiT's blocks on its unconditioned branch: the blocks set to identity
// while a step's fraction of the schedule lies in [Start, Stop). MOSS's negative branch skips block 22 over the whole
// schedule at no extra pass, and skip-layer guidance (arXiv 2403.17377) does the same over part of a schedule on any
// DiT, so it is data a guider carries rather than one model's special case.
type BlockOverride struct {
	Blocks      []int
	Start, Stop float64
}

// Validate refuses a block outside a model of layers blocks, and a span outside [0, 1] or reversed.
func (o BlockOverride) Validate(layers int) error {
	if !checked.UnitInterval64(o.Start) || !checked.UnitInterval64(o.Stop) || o.Start > o.Stop {
		return errors.New("guidance: a block override's span lies outside the schedule")
	}
	for _, block := range o.Blocks {
		if !checked.ValidIndex(block, layers) {
			return errors.New("guidance: a block override names a block the model does not have")
		}
	}
	return nil
}

// At is the blocks the override sets to identity at step of steps: its Blocks while the step's fraction of the schedule
// lies in its span, none otherwise.
func (o BlockOverride) At(step, steps int) []int {
	if fraction := float64(step) / float64(steps); len(o.Blocks) > 0 && o.Start <= fraction && fraction < o.Stop {
		return o.Blocks
	}
	return nil
}
