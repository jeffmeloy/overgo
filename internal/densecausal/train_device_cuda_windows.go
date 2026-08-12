//go:build windows

package densecausal

import (
	"fmt"

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
	return m.trainDevice(worker, steps, baseLR, mu, func() (float64, []float32, Grads, error) {
		return m.LossAndGrads(tokens)
	})
}

// TrainDeviceFull runs `steps` Muon updates with BOTH the backward and the Muon
// optimizer step on the GPU: deviceLossAndGrads (device layer backward via the
// resident MLP/attention blocks) plus DeviceMuonStepPlan. Only the forward and
// the head/norm/embedding tail run on host. Matches Train's host trajectory
// within fp32 tolerance. Attention bias is not yet supported (refused here at
// entry via DeviceTrainingSupported so an unsupported model fails before the
// resident session runs, not mid-step).
func (m *Model) TrainDeviceFull(worker *device.Worker, tokens []int, steps int, baseLR, mu float64) ([]float64, error) {
	if ok, reason := DeviceTrainingSupported(m.Dims); !ok {
		return nil, fmt.Errorf("TrainDeviceFull: %s", reason)
	}
	return m.trainDevice(worker, steps, baseLR, mu, func() (float64, []float32, Grads, error) {
		return m.deviceLossAndGrads(worker, tokens)
	})
}

func (m *Model) trainDevice(
	worker *device.Worker,
	steps int,
	baseLR, mu float64,
	lossAndGrads func() (float64, []float32, Grads, error),
) ([]float64, error) {
	names, weights, gradients, plan, resolvedLR, err := m.trainSetup(baseLR)
	if err != nil {
		return nil, err
	}
	config := optimizer.Config{BaseLearningRate: resolvedLR, Momentum: mu, Schedule: optimizer.ScheduleConstant}
	momentum := make([]float32, len(weights))

	trajectory := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		scatter(m, names, weights)
		loss, _, grads, err := lossAndGrads()
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
