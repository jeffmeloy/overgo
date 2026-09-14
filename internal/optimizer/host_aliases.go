// Package optimizer owns the device Muon steppers over the host optimizer core in hostoptimizer.
package optimizer

import "overgo/internal/hostoptimizer"

// The host optimizer core lives in hostoptimizer, which links no device
// runtime; the names below keep the device steppers and their callers on
// one vocabulary. A package that needs only the host core imports
// hostoptimizer directly and stays out of the device group.

// State is hostoptimizer.State.
type State = hostoptimizer.State

// StepResult is hostoptimizer.StepResult.
type StepResult = hostoptimizer.StepResult

// Optimizer is hostoptimizer.Optimizer.
type Optimizer = hostoptimizer.Optimizer

// GroupSpec is hostoptimizer.GroupSpec.
type GroupSpec = hostoptimizer.GroupSpec

// Group is hostoptimizer.Group.
type Group = hostoptimizer.Group

// Plan is hostoptimizer.Plan.
type Plan = hostoptimizer.Plan

// Config is hostoptimizer.Config.
type Config = hostoptimizer.Config

const (
	// ScheduleConstant is hostoptimizer.ScheduleConstant.
	ScheduleConstant = hostoptimizer.ScheduleConstant
	// ScheduleLinearDecay is hostoptimizer.ScheduleLinearDecay.
	ScheduleLinearDecay = hostoptimizer.ScheduleLinearDecay
)

// New is hostoptimizer.New.
func New(weights, gradients []float32, plan Plan, config Config) (*Optimizer, error) {
	return hostoptimizer.New(weights, gradients, plan, config)
}

// ValidateState is hostoptimizer.ValidateState.
func ValidateState(state State, planIdentity string, parameterCount int) error {
	return hostoptimizer.ValidateState(state, planIdentity, parameterCount)
}

// CompilePlan is hostoptimizer.CompilePlan.
func CompilePlan(parameterCount int, specs []GroupSpec) (Plan, error) {
	return hostoptimizer.CompilePlan(parameterCount, specs)
}
