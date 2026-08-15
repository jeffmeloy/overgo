package seq2seq

import "overgo/internal/optimizer"

func newTrainingOptimizer(weights, gradients []float32, plan optimizer.Plan, config optimizer.Config) (trainingOptimizer, error) {
	return optimizer.NewStepper(weights, gradients, plan, config)
}
