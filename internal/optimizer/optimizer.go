package optimizer

import (
	"errors"
	"fmt"
	"math"
)

// State: portable optimizer progress and momentum.
type State struct {
	PlanIdentity string
	Config       Config
	Step         int
	Momentum     []float64
}

// StepResult: committed step facts.
type StepResult struct {
	Step         int
	LearningRate float64
}

// Optimizer: flat CPU adaptive/Muon optimizer.
type Optimizer struct {
	weights   []float32
	gradients []float32
	plan      Plan
	config    Config
	momentum  []float64
	step      int
	scratch   newtonSchulzScratch
}

func New(weights, gradients []float32, plan Plan, config Config) (*Optimizer, error) {
	if plan.Identity() == "" {
		return nil, errors.New("optimizer: plan is not compiled")
	}
	if len(weights) != plan.ParameterCount() || len(gradients) != plan.ParameterCount() {
		return nil, fmt.Errorf("optimizer: weights and gradients must contain %d parameters", plan.ParameterCount())
	}
	if err := config.validate(); err != nil {
		return nil, err
	}
	optimizer := &Optimizer{
		weights:   weights,
		gradients: gradients,
		plan:      plan,
		config:    config,
		momentum:  make([]float64, plan.ParameterCount()),
	}
	optimizer.scratch.ensure(plan.maxMatrix, plan.maxSquare)
	return optimizer, nil
}

func (o *Optimizer) Step() StepResult {
	o.step++
	rate := o.config.LearningRate(o.step)
	for _, group := range o.plan.groups {
		if group.Frozen {
			clear(o.gradients[group.Start:group.End])
			continue
		}
		switch group.Update {
		case UpdateMuon:
			o.stepMuon(group, rate)
		case UpdateSign:
			o.stepSign(group, rate)
		}
	}
	return StepResult{Step: o.step, LearningRate: rate}
}

func (o *Optimizer) stepSign(group Group, rate float64) {
	for index := group.Start; index < group.End; index++ {
		gradient := float64(o.gradients[index])
		momentum, direction := nesterov(o.config.Momentum, o.momentum[index], gradient)
		o.momentum[index] = momentum
		if gradient != 0 {
			switch {
			case direction > 0:
				o.weights[index] -= float32(rate)
			case direction < 0:
				o.weights[index] += float32(rate)
			}
		}
		o.gradients[index] = 0
	}
}

func (o *Optimizer) stepMuon(group Group, rate float64) {
	length := group.End - group.Start
	direction := o.scratch.input[:length]
	for offset := range length {
		index := group.Start + offset
		momentum, nesterovDirection := nesterov(o.config.Momentum, o.momentum[index], float64(o.gradients[index]))
		o.momentum[index] = momentum
		direction[offset] = nesterovDirection
	}
	newtonSchulz(direction, group.Rows, group.Cols, &o.scratch)
	scale := math.Sqrt(float64(max(group.Rows, group.Cols))) * stepRMS(o.config.Momentum) * rate
	for offset := range length {
		index := group.Start + offset
		o.weights[index] -= float32(scale * direction[offset])
		o.gradients[index] = 0
	}
}

func nesterov(momentum, previous, gradient float64) (next, direction float64) {
	next = momentum*previous + gradient
	return next, momentum*next + gradient
}

func (o *Optimizer) Snapshot() State {
	return State{
		PlanIdentity: o.plan.Identity(),
		Config:       o.config,
		Step:         o.step,
		Momentum:     append([]float64(nil), o.momentum...),
	}
}

func (o *Optimizer) Restore(state State) error {
	if state.PlanIdentity != o.plan.Identity() {
		return errors.New("optimizer state: plan identity mismatch")
	}
	if state.Config != o.config {
		return errors.New("optimizer state: config mismatch")
	}
	if state.Step < 0 {
		return errors.New("optimizer state: negative step")
	}
	if len(state.Momentum) != len(o.momentum) {
		return errors.New("optimizer state: momentum length mismatch")
	}
	for _, value := range state.Momentum {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return errors.New("optimizer state: non-finite momentum")
		}
	}
	o.step = state.Step
	copy(o.momentum, state.Momentum)
	return nil
}
