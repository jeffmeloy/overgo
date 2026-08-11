package densecausal

import (
	"fmt"
	"sort"

	"overgo/internal/optimizer"
)

// Train runs `steps` full-parameter Muon updates over one token batch and
// returns the loss trajectory (entry k is the loss measured before update k, so
// trajectory[0] is the pre-training loss). Every trainable tensor flattens into
// one optimizer plan; the Muon geometry is derived from each tensor's stored
// shape -- 2D tensors get the Newton-Schulz matrix rule, 1D tensors the vector
// rule -- never named per family. A non-positive baseLR is DERIVED from the
// trainable parameter count (optimizer.DeriveBaseLR: n_params^-1/2), so one
// call trains any scale without a hand-tuned rate; pass a positive baseLR only
// to override. This is the host learning loop; the device path is a later rung
// that must match this trajectory within tolerance.
func (m *Model) Train(tokens []int, steps int, baseLR, mu float64) ([]float64, error) {
	names := make([]string, 0, len(m.Weights))
	for name := range m.Weights {
		names = append(names, name)
	}
	sort.Strings(names)

	total := 0
	for _, name := range names {
		total += len(m.Weights[name])
	}
	if baseLR <= 0 {
		baseLR = optimizer.DeriveBaseLR(total)
	}
	weights := make([]float32, total)
	gradients := make([]float32, total)
	specs := make([]optimizer.GroupSpec, 0, len(names))
	offset := 0
	for _, name := range names {
		values := m.Weights[name]
		rows, cols, err := tensorGeometry(m.Shapes[name], len(values))
		if err != nil {
			return nil, fmt.Errorf("densecausal: %s: %w", name, err)
		}
		copy(weights[offset:], values)
		specs = append(specs, optimizer.GroupSpec{
			Name: name, Start: offset, End: offset + len(values), Rows: rows, Cols: cols,
		})
		offset += len(values)
	}
	plan, err := optimizer.CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	opt, err := optimizer.New(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: baseLR, Momentum: mu, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, err
	}

	trajectory := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		// Scatter optimizer weights into the model before each forward pass.
		scatter(m, names, weights)
		loss, _, grads, err := m.LossAndGrads(tokens)
		if err != nil {
			return nil, err
		}
		trajectory = append(trajectory, loss)
		offset = 0
		for _, name := range names {
			n := len(m.Weights[name])
			if g, ok := grads[name]; ok {
				copy(gradients[offset:offset+n], g)
			} else {
				clear(gradients[offset : offset+n])
			}
			offset += n
		}
		opt.Step()
	}
	// Final scatter so the model reflects the last update.
	scatter(m, names, weights)
	return trajectory, nil
}

// scatter copies the flat optimizer weight buffer back into the model tensors.
func scatter(m *Model, names []string, weights []float32) {
	offset := 0
	for _, name := range names {
		n := len(m.Weights[name])
		copy(m.Weights[name], weights[offset:offset+n])
		offset += n
	}
}

// tensorGeometry maps a stored tensor shape to Muon (rows, cols): a 2D tensor
// keeps its shape (Newton-Schulz matrix geometry), a 1D tensor is a
// (length, 1) vector; any other rank is an unsupported training layout.
func tensorGeometry(shape []int, length int) (int, int, error) {
	switch len(shape) {
	case 2:
		if shape[0]*shape[1] != length {
			return 0, 0, fmt.Errorf("shape %v does not match length %d", shape, length)
		}
		return shape[0], shape[1], nil
	case 1:
		if shape[0] != length {
			return 0, 0, fmt.Errorf("shape %v does not match length %d", shape, length)
		}
		return shape[0], 1, nil
	default:
		return 0, 0, fmt.Errorf("unsupported tensor rank %d (shape %v)", len(shape), shape)
	}
}
