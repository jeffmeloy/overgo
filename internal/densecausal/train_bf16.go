package densecausal

import "overgo/internal/optimizer"

// TrainBF16SGD trains with the at-scale memory path: a BF16 master (2 bytes/
// param, half an F32 master) updated by plain stochastic-rounding SGD, with NO
// momentum state -- the optimizer footprint a model too large for Muon/F32-grad
// needs. Gradients are computed in F32 (LossAndGrads) and consumed per step; the
// persistent optimizer state is the BF16 master alone. Returns the pre-update
// loss trajectory. Non-positive lr derives from parameter count.
//
// This is the host demonstration of the optimizer half of train-scale-bf16-sgd;
// the residency win at full scale additionally requires the forward to read the
// BF16 master directly (native-dtype) rather than an F32 working copy.
func (m *Model) TrainBF16SGD(tokens []int, steps int, lr float64, seed uint64) ([]float64, error) {
	names, weights, gradients, _, resolvedLR, err := m.trainSetup(lr)
	if err != nil {
		return nil, err
	}
	opt := optimizer.NewBF16SGD(weights, seed)
	work := make([]float32, len(weights))
	trajectory := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		opt.WeightsInto(work)
		scatter(m, names, work)
		loss, _, grads, err := m.LossAndGrads(tokens)
		if err != nil {
			return nil, err
		}
		trajectory = append(trajectory, loss)
		m.gatherGrads(names, gradients, grads)
		opt.Step(gradients, resolvedLR)
	}
	opt.WeightsInto(work)
	scatter(m, names, work)
	return trajectory, nil
}
