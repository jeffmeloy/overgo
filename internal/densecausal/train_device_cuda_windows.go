//go:build windows

package densecausal

import (
	"overgo/internal/cuda/device"
	"overgo/internal/optimizer"
)

// TrainDevice runs `steps` Muon updates over one token batch with the optimizer
// step on the GPU (device Newton-Schulz), matching Train's host trajectory
// within fp32 tolerance. Forward and backward (LossAndGrads) still run on the
// host; only the Muon optimizer step -- whose host Newton-Schulz was the scaling
// bottleneck -- is device-accelerated. Momentum is fp32 device-native. Returns
// the loss trajectory (entry k is the loss before update k).
func (m *Model) TrainDevice(worker *device.Worker, tokens []int, steps int, baseLR, mu float64) ([]float64, error) {
	names, weights, gradients, plan, resolvedLR, err := m.trainSetup(baseLR)
	if err != nil {
		return nil, err
	}
	config := optimizer.Config{BaseLearningRate: resolvedLR, Momentum: mu, Schedule: optimizer.ScheduleConstant}
	momentum := make([]float32, len(weights))

	trajectory := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		scatter(m, names, weights)
		loss, _, grads, err := m.LossAndGrads(tokens)
		if err != nil {
			return nil, err
		}
		trajectory = append(trajectory, loss)
		m.gatherGrads(names, gradients, grads)
		if err := optimizer.DeviceMuonStepPlan(worker, weights, gradients, momentum, plan, step+1, config); err != nil {
			return nil, err
		}
	}
	scatter(m, names, weights)
	return trajectory, nil
}
