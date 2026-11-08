package sampling

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/checked"
)

// FlowSampler walks a sample down a sigma table, a knot more than its steps, the last the one the walk ends at: Step
// moves the sample from the knot it is at to the next by the guided model output there. Wan's flow UniPC, the image
// models' flow Euler, RAE's Euler and SimpleDiffusion's are its implementations, so a schedule or a step rule is
// written once, and Walk is every model's loop. A sampler holds its place on the table, so one walks one sample once.
type FlowSampler interface {
	Sigmas() []float64
	Step(output, sample, modelOutput []float32) error
}

// Noise fills a sample with the noise a walk starts from.
type Noise interface {
	Fill(sample []float32) error
}

// Fill draws the plan's noise into sample (FillCounterNormalNoise).
func (plan CounterNoisePlan) Fill(sample []float32) error {
	return FillCounterNormalNoise(sample, plan)
}

// Walk runs sampler down its table from sample, the model giving its output at the sample of each step: the sample the
// last step reaches is returned, in one of the two buffers the walk alternates between, sample among them.
func Walk(ctx context.Context, sampler FlowSampler, sample []float32, model func(step int, sample []float32) ([]float32, error)) ([]float32, error) {
	if sampler == nil || model == nil {
		return nil, errors.New("sampling: a walk needs a sampler and a model")
	}
	next := make([]float32, len(sample))
	for step := range len(sampler.Sigmas()) - 1 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		output, err := model(step, sample)
		if err != nil {
			return nil, err
		}
		if err := sampler.Step(next, sample, output); err != nil {
			return nil, err
		}
		sample, next = next, sample
	}
	return sample, nil
}

// Sigmas is the UniPC schedule's table, its terminal zero last.
func (p *UniPCSampler) Sigmas() []float64 {
	sigmas := make([]float64, len(p.sigmas))
	for index, sigma := range p.sigmas {
		sigmas[index] = float64(sigma)
	}
	return sigmas
}

// FlowEuler steps a flow sample along its table by the model's velocity times the step's delta, in float32 as the
// references do. The deltas are the caller's, each rounded as its reference rounds it: the change of a float64 table
// rounded once (the image models' dynamic shift), the change of a float32 table (RAE's grid, the nearest-grid
// schedule), or one fixed interval (SimpleDiffusion).
type FlowEuler struct {
	sigmas []float64
	deltas []float32
	step   int
}

// NewFlowEuler is the Euler walk of a table and its steps' deltas, one fewer than the knots.
func NewFlowEuler(sigmas []float64, deltas []float32) (*FlowEuler, error) {
	if len(deltas) == 0 || !checked.Equal(len(deltas), len(sigmas)-1) {
		return nil, fmt.Errorf("sampling: a flow Euler walk of %d knots has %d deltas", len(sigmas), len(deltas))
	}
	for _, sigma := range sigmas {
		if !checked.Finite64(sigma) {
			return nil, errors.New("sampling: a flow Euler table is not finite")
		}
	}
	for _, delta := range deltas {
		if !checked.Finite32(delta) {
			return nil, errors.New("sampling: a flow Euler delta is not finite")
		}
	}
	return &FlowEuler{sigmas: append([]float64(nil), sigmas...), deltas: append([]float32(nil), deltas...)}, nil
}

// Sigmas is the walk's table.
func (e *FlowEuler) Sigmas() []float64 { return e.sigmas }

// Delta is the step's delta, which a device's own update reads where the step runs on the card.
func (e *FlowEuler) Delta(step int) float32 { return e.deltas[step] }

// Step writes sample plus the step's delta times the model's output into output, and moves to the next knot.
func (e *FlowEuler) Step(output, sample, modelOutput []float32) error {
	if !checked.ValidIndex(e.step, len(e.deltas)) {
		return fmt.Errorf("sampling: flow Euler step %d past its table", e.step)
	}
	if !checked.Equal(len(output), len(sample)) || !checked.Equal(len(modelOutput), len(sample)) {
		return fmt.Errorf("sampling: flow Euler buffers %d/%d/%d differ", len(output), len(sample), len(modelOutput))
	}
	delta := e.deltas[e.step]
	for index, value := range sample {
		output[index] = value + delta*modelOutput[index]
	}
	e.step++
	return nil
}

// Guider combines a model's conditional and unconditional outputs into the one a sampler steps by: the unconditional
// plus Scale times their difference. Before sees the two outputs before the combine and After the guided one after it,
// either free to change them, so a guidance method is one function. A reference that guides in float32 (RAE) combines
// in float32; the others in float64, rounded once.
type Guider struct {
	Scale   float64
	Float32 bool
	Before  func(conditional, unconditional []float32) error
	After   func(guided []float32) error
	// Override is what the unconditioned branch changes of the model's blocks (BlockOverride).
	Override BlockOverride
}

// unguided is the scale at which the guided output is the conditional one.
const unguided = 1

// Unconditional says whether the guider reads an unconditional output: at a scale of one the guided output is the
// conditional one, so a model runs no unconditioned pass for it.
func (g Guider) Unconditional() bool { return g.Scale != unguided }

// Combine writes the guided output into out, which may be either output it reads: each element is read before it is
// written. A guider that reads no unconditional output passes the conditional one through, and is given none.
func (g Guider) Combine(out, conditional, unconditional []float32) error {
	if g.Before != nil {
		if err := g.Before(conditional, unconditional); err != nil {
			return err
		}
	}
	if !g.Unconditional() {
		if !checked.Equal(len(out), len(conditional)) {
			return fmt.Errorf("guidance: buffer lengths %d/%d differ", len(out), len(conditional))
		}
		copy(out, conditional)
	} else if g.Float32 {
		if !checked.Equal(len(out), len(conditional)) || !checked.Equal(len(out), len(unconditional)) {
			return fmt.Errorf("guidance: buffer lengths %d/%d/%d differ", len(out), len(conditional), len(unconditional))
		}
		scale := float32(g.Scale)
		for index, value := range unconditional {
			out[index] = value + scale*(conditional[index]-value)
		}
	} else if err := GuideInto(out, conditional, unconditional, g.Scale); err != nil {
		return err
	}
	if g.After != nil {
		return g.After(out)
	}
	return nil
}
