package optimizer

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
)

// State: portable optimizer progress and momentum.
type State struct {
	PlanIdentity string    `json:"plan_identity"`
	Config       Config    `json:"config"`
	Step         int       `json:"step"`
	Momentum     []float64 `json:"momentum"`
}

// StepResult: committed step facts.
type StepResult struct {
	Step         int
	LearningRate float64
	GradientL2   float64
	UpdateL2     float64
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
	return optimizer, nil
}

func (o *Optimizer) Step() StepResult {
	return o.StepGroups(func(Group) bool { return true })
}

// StepGroups: one step over selected groups; excluded state remains unchanged.
func (o *Optimizer) StepGroups(include func(Group) bool) StepResult {
	o.step++
	rate := o.config.LearningRate(o.step)
	var gradientSquared, updateSquared float64
	for _, group := range o.plan.groups {
		if !include(group) {
			continue
		}
		if group.Frozen {
			clear(o.gradients[group.Start:group.End])
			continue
		}
		gradient, update := o.stepMuon(group, rate)
		gradientSquared += gradient
		updateSquared += update
	}
	return StepResult{Step: o.step, LearningRate: rate, GradientL2: math.Sqrt(gradientSquared), UpdateL2: math.Sqrt(updateSquared)}
}

func (o *Optimizer) stepMuon(group Group, rate float64) (gradientSquared, updateSquared float64) {
	length := group.End - group.Start
	o.scratch.ensure(o.plan.maxMatrix, o.plan.maxSquare)
	direction := o.scratch.input[:length]
	for offset := range length {
		index := group.Start + offset
		gradient := float64(o.gradients[index])
		gradientSquared += gradient * gradient
		momentum, nesterovDirection := nesterov(o.config.Momentum, o.momentum[index], gradient)
		o.momentum[index] = momentum
		direction[offset] = nesterovDirection
	}
	newtonSchulz(direction, group.Rows, group.Cols, &o.scratch)
	scale := math.Sqrt(float64(max(group.Rows, group.Cols))) * stepRMS(o.config.Momentum) * rate
	for offset := range length {
		index := group.Start + offset
		update := scale * direction[offset]
		updateSquared += update * update
		o.weights[index] -= float32(update)
		o.gradients[index] = 0
	}
	return gradientSquared, updateSquared
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
	if err := ValidateState(state, o.plan.Identity(), len(o.momentum)); err != nil {
		return err
	}
	if state.Config != o.config {
		return errors.New("optimizer state: config mismatch")
	}
	o.step = state.Step
	copy(o.momentum, state.Momentum)
	return nil
}

// ValidateState checks portable Muon state without live parameter storage.
func ValidateState(state State, planIdentity string, parameterCount int) error {
	if planIdentity == "" || state.PlanIdentity != planIdentity {
		return errors.New("optimizer state: plan identity mismatch")
	}
	identity, err := hex.DecodeString(state.PlanIdentity)
	if err != nil || len(identity) != sha256.Size {
		return errors.New("optimizer state: invalid plan identity")
	}
	if err := state.Config.validate(); err != nil {
		return err
	}
	if state.Step < 0 {
		return errors.New("optimizer state: negative step")
	}
	if parameterCount < 0 || len(state.Momentum) != parameterCount {
		return errors.New("optimizer state: momentum length mismatch")
	}
	for _, value := range state.Momentum {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return errors.New("optimizer state: non-finite momentum")
		}
	}
	return nil
}
