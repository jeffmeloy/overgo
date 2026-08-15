//go:build windows

package adaptertrain

import (
	"errors"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

type stepState struct {
	model   *Model
	worker  *device.Worker
	example Example
	output  []float32
	trace   hostmath.PerLayerAdapterTrace
	loss    float64
	dOutput []float32
}

// Step executes the compiled forward/loss/backward/Muon program.
func (m *Model) Step(worker *device.Worker, example Example) (float64, error) {
	if worker == nil {
		return 0, errors.New("adapter training: worker unavailable")
	}
	if err := m.validateExample(example); err != nil {
		return 0, err
	}
	execution, err := trainingprogram.Bind(m.program, []trainingprogram.Binding[stepState]{
		{Operator: "adapter-forward", Execute: func(state *stepState) error {
			output, trace, err := hostmath.PerLayerAdapterForward(
				state.example.Input, state.example.Side, m.gate, m.projection, m.norm,
				state.example.Rows, m.hidden, m.width, m.epsilon,
			)
			state.output, state.trace = output, trace
			return err
		}},
		{Operator: "squared-error", Execute: func(state *stepState) error {
			state.dOutput = make([]float32, len(state.output))
			inverse := 1 / float64(len(state.output))
			for index, value := range state.output {
				delta := float64(value) - float64(state.example.Target[index])
				state.loss += 0.5 * delta * delta * inverse
				state.dOutput[index] = float32(delta * inverse)
			}
			if math.IsNaN(state.loss) || math.IsInf(state.loss, 0) {
				return errors.New("adapter training: non-finite loss")
			}
			return nil
		}},
		{Operator: "adapter-backward", Execute: func(state *stepState) error {
			_, _, gate, projection, norm, err := hostmath.PerLayerAdapterBackward(
				state.example.Input, state.example.Side, m.gate, m.projection, m.norm, state.dOutput,
				state.example.Rows, m.hidden, m.width, m.epsilon, state.trace,
			)
			if err != nil {
				return err
			}
			copy(m.gradients, gate)
			copy(m.gradients[len(gate):], projection)
			copy(m.gradients[len(gate)+len(projection):], norm)
			return nil
		}},
		{Operator: "muon", Execute: func(state *stepState) error {
			next := m.step + 1
			if err := optimizer.DeviceMuonStepPlan(state.worker, m.weights, m.gradients, m.momentum, m.plan, next, m.config); err != nil {
				return err
			}
			m.step = next
			return nil
		}},
	})
	if err != nil {
		return 0, err
	}
	state := stepState{model: m, worker: worker, example: example}
	if err := execution.RunPhases(&state,
		trainingprogram.PhaseForward, trainingprogram.PhaseLoss,
		trainingprogram.PhaseBackward, trainingprogram.PhaseOptimize,
	); err != nil {
		return 0, err
	}
	return state.loss, nil
}
