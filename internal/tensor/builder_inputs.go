package tensor

import "overgo/internal/tensor/dtype"

// WeightInputs: named graph weights with compiled matrix storage.
type WeightInputs struct {
	Builder    *Builder
	Inputs     map[string]*Tensor
	MatrixType dtype.Type
}

func (b WeightInputs) Input(name string, dimensions ...uint64) *Tensor {
	storage := dtype.F32
	if len(dimensions) == 2 {
		storage = b.MatrixType
	}
	node := b.Builder.Input(name, storage, MustShape(dimensions...))
	b.Inputs[name] = node
	return node
}
