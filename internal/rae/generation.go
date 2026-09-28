package rae

import (
	"fmt"
	"math"
	"slices"
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

// timeGrid is the reference's shifted sampling grid of steps points.
func (m *diffusionTransformer) timeGrid(steps int) []float32 {
	shift := math.Sqrt(float64(m.latent*m.grid*m.grid) / shiftBase)
	grid := make([]float32, steps)
	for index := range steps {
		t := 1 - lastTime*float64(index)/float64(steps-1)
		grid[index] = float32(shift * t / (1 + (shift-1)*t))
	}
	return grid
}

// generate samples a class from channel-first noise: the DiT's velocity
// integrated by Euler over the shifted grid of steps points, the latent it
// reaches decoded to pixels.
func generate(model *diffusionTransformer, decoder *vitDecoder, noise []float32, label, steps int) ([]float32, error) {
	if steps < minimumSteps {
		return nil, fmt.Errorf("rae: a sample needs at least %d grid points, got %d", minimumSteps, steps)
	}
	grid := model.timeGrid(steps)
	latent := slices.Clone(noise)
	for index := range steps - 1 {
		velocity, err := model.velocity(latent, grid[index], label)
		if err != nil {
			return nil, err
		}
		step := grid[index+1] - grid[index]
		for position, value := range velocity {
			latent[position] += step * value
		}
	}
	return decoder.decode(latent)
}
