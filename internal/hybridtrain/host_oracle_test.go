package hybridtrain

import (
	"errors"

	"overgo/internal/optimizer"
)

type hostTraining struct {
	hostBackwardPass
	matOpt, vecOpt *optimizer.Optimizer
}

func (training *hostTraining) Step(_ int) error {
	training.matOpt.Step()
	training.vecOpt.Step()
	return nil
}

// TrainHost runs the host parity lane.
func (m *Model) TrainHost(steps int, cfg optimizer.Config, options TrainingOptions) ([]float64, error) {
	mat, vec, err := m.resumeParts(steps, cfg, options, false)
	if err != nil {
		return nil, err
	}
	training := &hostTraining{hostBackwardPass: newHostBackwardPass(m)}
	training.matOpt, err = optimizer.New(m.matW, training.matGrad, m.matPlan, cfg)
	if err != nil {
		return nil, err
	}
	training.vecOpt, err = optimizer.New(m.vecW, training.vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, err
	}
	if err := errors.Join(restoreOptimizer(training.matOpt, mat), restoreOptimizer(training.vecOpt, vec)); err != nil {
		return nil, err
	}
	trajectory, err := runTrainingObserved(m.program, training, m.Target, steps, mat.Step, options.Observe)
	if trajectory != nil && options.Checkpoint != nil {
		err = errors.Join(err, m.checkpoint(options, training.matOpt.Snapshot(), training.vecOpt))
	}
	return trajectory, err
}
