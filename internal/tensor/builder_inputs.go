package tensor

import (
	"fmt"

	"overgo/internal/tensor/dtype"
)

// WeightBindings preserve graph compilation order.
type WeightBindings []*Tensor

// InputBinding pairs one graph input with its external value.
type InputBinding[T any] struct {
	Node  *Tensor
	Value T
}

// InputBindings preserve graph construction order.
type InputBindings[T any] []InputBinding[T]

func (b *InputBindings[T]) Add(node *Tensor, value T) {
	*b = append(*b, InputBinding[T]{Node: node, Value: value})
}

// Node resolves one compiled binding.
func (b WeightBindings) Node(name string) *Tensor {
	for _, node := range b {
		if node.Name == name {
			return node
		}
	}
	return nil
}

// WeightInputs records ordered graph weights with matrix storage.
type WeightInputs struct {
	Builder    *Builder
	Bindings   *WeightBindings
	MatrixType dtype.Type
}

func (b WeightInputs) Input(name string, dimensions ...uint64) *Tensor {
	if existing := b.Bindings.Node(name); existing != nil {
		b.Builder.setError(fmt.Errorf("tensor weight binding %s is duplicated", name))
		return existing
	}
	storage := dtype.F32
	if len(dimensions) == 2 {
		storage = b.MatrixType
	}
	node := b.Builder.Input(name, storage, MustShape(dimensions...))
	*b.Bindings = append(*b.Bindings, node)
	return node
}
