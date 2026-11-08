package rae

import (
	"context"
	"fmt"
	"math"
	"slices"

	"overgo/internal/sampling"
)

// The reference's sampling grid runs from pure noise at one to a thousandth
// short of the data, shifted toward noise by the square root of the latent's
// size over the size its schedule was made for.
const (
	lastTime  = 0.999
	shiftBase = 4096
)

// minimumSteps is the fewest grid points a sample needs: a start and an end.
const minimumSteps = 2

// shift is the reference's timestep shift for the DiT's latent: the square root of its size over shiftBase.
func (m *diffusionTransformer) shift() float64 {
	return math.Sqrt(float64(m.latent*m.grid*m.grid) / shiftBase)
}

// timeGrid is the reference's shifted sampling grid of steps points.
func (m *diffusionTransformer) timeGrid(steps int) []float32 {
	shift := m.shift()
	grid := make([]float32, steps)
	for index := range steps {
		t := 1 - lastTime*float64(index)/float64(steps-1)
		grid[index] = float32(shift * t / (1 + (shift-1)*t))
	}
	return grid
}

// sample integrates the DiT's velocity for a class from channel-first noise
// by Euler over the shifted grid of steps points and returns the latent it
// reaches, each step's delta the change of the float32 grid. Above a guidance
// scale of one, each step's velocity is pushed away from the null class's by
// that scale in float32, as the reference's forward_with_cfg does; at one the
// class velocity alone drives the step.
func (m *diffusionTransformer) sample(ctx context.Context, noise []float32, label, steps int, guidance float32) ([]float32, error) {
	if steps < minimumSteps {
		return nil, fmt.Errorf("rae: a sample needs at least %d grid points, got %d", minimumSteps, steps)
	}
	grid := m.timeGrid(steps)
	knots := make([]float64, steps)
	deltas := make([]float32, steps-1)
	for index, knot := range grid {
		knots[index] = float64(knot)
		if index > 0 {
			deltas[index-1] = knot - grid[index-1]
		}
	}
	sampler, err := sampling.NewFlowEuler(knots, deltas)
	if err != nil {
		return nil, err
	}
	guider := sampling.Guider{Scale: float64(guidance), Float32: true}
	return sampling.Walk(ctx, sampler, slices.Clone(noise), func(index int, latent []float32) ([]float32, error) {
		velocity, err := m.velocity(latent, grid[index], label)
		if err != nil || guidance <= 1 {
			return velocity, err
		}
		unconditional, err := m.velocity(latent, grid[index], m.classes)
		if err != nil {
			return nil, err
		}
		return velocity, guider.Combine(velocity, velocity, unconditional)
	})
}
