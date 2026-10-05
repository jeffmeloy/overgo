package densecausal

import (
	"errors"
	"fmt"
	"strings"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// A model that holds weights frozen in their stored quantization keeps each such
// projection as the bytes a GGUF file holds it in and dequantizes it into one
// scratch buffer per kind of projection, shared by every layer, just before the
// layer reads it, so the model's resident weights are the quantized bytes, the
// full-precision tensors that are never adapted, and one layer's worth of scratch.
// It is for training an adapter over the weights, which never change; its
// scratch is not safe to use from two goroutines, and the model trains no
// weight of its own.

// frozenTensor is one frozen projection as its file stores it.
type frozenTensor struct {
	stored dtype.Type
	data   []byte
}

// frozenWeights are a model's frozen projections, the scratch they dequantize
// into, and the adjustment applied to each dequantized weight.
type frozenWeights struct {
	tensors map[string]frozenTensor
	// scratch is one buffer per kind of projection (self_attn.q_proj.weight), sized for
	// the largest layer's, which every layer's slice of that kind is a prefix of.
	scratch map[string][]float32
	adjust  func(name string, weights []float32) error
}

// projectionKind is a weight name without its layer: the kind of projection
// every layer's has, and the key of its scratch.
func projectionKind(name string) string {
	rest, found := strings.CutPrefix(name, "model.layers.")
	if !found {
		return name
	}
	if _, kind, found := strings.Cut(rest, "."); found {
		return kind
	}
	return name
}

// Frozen reports whether the weight is held frozen in its stored quantization.
func (m *Model) Frozen(name string) bool {
	if m == nil || m.frozen == nil {
		return false
	}
	_, held := m.frozen.tensors[name]
	return held
}

// OnDequantize sets the adjustment applied, in place, to each frozen weight
// right after it is dequantized for a layer to read, and nil removes it: an
// adapter adds its update here, so the layer reads the base plus the update
// without either being held beyond the layer that reads it.
func (m *Model) OnDequantize(adjust func(name string, weights []float32) error) error {
	if m == nil || m.frozen == nil {
		return errors.New("densecausal: the model holds no frozen weights")
	}
	m.frozen.adjust = adjust
	return nil
}

// WeightBytes are the bytes the model's weights occupy: the full-precision
// tensors it holds, and, for a model with frozen weights, the quantized bytes
// and the scratch they dequantize into.
func (m *Model) WeightBytes() (full, quantized, scratch int) {
	for _, values := range m.Weights {
		full += len(values) * float32Bytes
	}
	if m.frozen != nil {
		for _, tensor := range m.frozen.tensors {
			quantized += len(tensor.data)
		}
		for _, buffer := range m.frozen.scratch {
			scratch += len(buffer) * float32Bytes
		}
	}
	return full, quantized, scratch
}

// float32Bytes is the width of one full-precision weight.
const float32Bytes = 4

// useLayer is layer index with its frozen projections dequantized into the
// scratch the layer's slices alias, and adjusted: what a layer reads.
func (m *Model) useLayer(index int) (layer, error) {
	l := m.layers[index]
	if m.frozen == nil {
		return l, nil
	}
	names := l.names
	for _, name := range []string{names.Q, names.K, names.V, names.O, names.Gate, names.Up, names.Down} {
		tensor, held := m.frozen.tensors[name]
		if !held {
			continue
		}
		weights := m.frozen.scratch[projectionKind(name)][:m.elements(name)]
		if err := quant.DequantizeInto(tensor.stored, tensor.data, weights); err != nil {
			return layer{}, fmt.Errorf("densecausal: dequantize %s: %w", name, err)
		}
		if m.frozen.adjust != nil {
			if err := m.frozen.adjust(name, weights); err != nil {
				return layer{}, err
			}
		}
	}
	return l, nil
}

// elements is how many weights the named tensor holds.
func (m *Model) elements(name string) int {
	total := 1
	for _, extent := range m.Shapes[name] {
		total *= extent
	}
	return total
}

// errFrozen refuses training a model whose weights are frozen.
var errFrozen = errors.New("densecausal: a model with frozen weights trains only an adapter over them")
