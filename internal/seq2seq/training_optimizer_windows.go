//go:build windows

package seq2seq

import (
	"overgo/internal/cuda/device"
	"overgo/internal/optimizer"
)

type deviceTrainingOptimizer struct {
	worker             *device.Worker
	weights, gradients []float32
	momentum           []float32
	plan               optimizer.Plan
	config             optimizer.Config
	step               int
}

func newTrainingOptimizer(weights, gradients []float32, plan optimizer.Plan, config optimizer.Config) (trainingOptimizer, error) {
	worker, err := device.New(0)
	if err != nil {
		return nil, err
	}
	return &deviceTrainingOptimizer{
		worker: worker, weights: weights, gradients: gradients, momentum: make([]float32, len(weights)),
		plan: plan, config: config,
	}, nil
}

func (o *deviceTrainingOptimizer) Step() error {
	o.step++
	return optimizer.DeviceMuonStepPlan(o.worker, o.weights, o.gradients, o.momentum, o.plan, o.step, o.config)
}

func (o *deviceTrainingOptimizer) Close() error { return o.worker.Close() }
