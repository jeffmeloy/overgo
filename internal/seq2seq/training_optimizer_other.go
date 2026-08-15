//go:build !windows

package seq2seq

import "overgo/internal/optimizer"

type hostTrainingOptimizer struct{ optimizer *optimizer.Optimizer }

func newTrainingOptimizer(weights, gradients []float32, plan optimizer.Plan, config optimizer.Config) (trainingOptimizer, error) {
	instance, err := optimizer.New(weights, gradients, plan, config)
	if err != nil {
		return nil, err
	}
	return &hostTrainingOptimizer{optimizer: instance}, nil
}

func (o *hostTrainingOptimizer) Step() error {
	o.optimizer.Step()
	return nil
}

func (o *hostTrainingOptimizer) Close() error { return nil }
