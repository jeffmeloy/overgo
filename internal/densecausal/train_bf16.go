package densecausal

import (
	"overgo/internal/optimizer"
	"overgo/internal/tensor/dtype"
)

// TrainBF16SGD trains with the bf16-SGD memory path: a bf16 master (2 bytes/
// param, half an fp32 master) updated by plain stochastic-rounding SGD, with NO
// momentum state. The bf16 master is the SOLE persistent weight store -- the flat
// fp32 weight buffer is released after seeding it, and each step upconverts the
// master into the model's existing fp32 weight slices in place (scatterBF16), so
// no separate fp32 work slab is held. Persistent state is therefore
// Model.Weights(fp32, reused) + gradients(fp32) + master(bf16); versus the old
// path this drops a flat fp32 weight copy and a full-model fp32 work slab.
//
// This is the optimizer + host reduction. Driving Model.Weights itself below fp32
// (forward reading the bf16 master per-layer, no full fp32 copy) is the remaining
// native-bf16-forward step. Returns the pre-update loss trajectory. Non-positive
// lr derives from parameter count.
func (m *Model) TrainBF16SGD(tokens []int, steps int, lr float64, seed uint64) ([]float64, error) {
	names, weights, gradients, _, resolvedLR, err := m.trainSetup(lr)
	if err != nil {
		return nil, err
	}
	opt := optimizer.NewBF16SGD(weights, seed)
	weights = nil // release the flat fp32 weights; the bf16 master is now the sole store
	_ = weights
	trajectory := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		scatterBF16(m, names, opt.Master())
		loss, _, grads, err := m.LossAndGrads(tokens)
		if err != nil {
			return nil, err
		}
		trajectory = append(trajectory, loss)
		m.gatherGrads(names, gradients, grads)
		opt.Step(gradients, resolvedLR)
	}
	scatterBF16(m, names, opt.Master())
	return trajectory, nil
}

// scatterBF16 upconverts the flat bf16 master into the model's fp32 weight slices
// in place (names order, matching trainSetup's flat layout), holding no separate
// fp32 work slab.
func scatterBF16(m *Model, names []string, master []uint16) {
	offset := 0
	for _, name := range names {
		w := m.Weights[name]
		for j := range w {
			w[j] = dtype.BF16ToFloat32(master[offset+j])
		}
		offset += len(w)
	}
}
