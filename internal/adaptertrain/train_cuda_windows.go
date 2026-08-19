//go:build windows

package adaptertrain

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/optimizer"
)

type deviceOptimizer struct{ worker *device.Worker }

func (update deviceOptimizer) Step(model *Model) error {
	next := model.step + 1
	if err := optimizer.DeviceMuonStepPlan(update.worker, model.weights, model.gradients, model.momentum, model.plan, next, model.config); err != nil {
		return err
	}
	model.step = next
	return nil
}

// Step executes the compiled forward/loss/backward/Muon program.
func (m *Model) Step(worker *device.Worker, example Example) (float64, error) {
	if worker == nil {
		return 0, errors.New("adapter training: worker unavailable")
	}
	if err := m.validateExample(example); err != nil {
		return 0, err
	}
	state := stepState{model: m, optimizer: deviceOptimizer{worker}, example: example}
	if err := m.execution.Run(&state); err != nil {
		return 0, err
	}
	return state.loss, nil
}
