package inference

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

type layerInputCapture struct {
	order     []int32
	requested map[int32]struct{}
	values    map[int32]reference.Value
}

// ForwardCachedExtractLayerInputs: cached forward plus pre-layer rows.
func (r *Runner) ForwardCachedExtractLayerInputs(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
	layerIDs []int32,
) (reference.Value, *KVCache, reference.Value, error) {
	if r == nil {
		return reference.Value{}, nil, reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, reference.Value{}, errors.New("inference: runner is closed")
	}
	if r.spec.NonCausalAttention || r.spec.Architecture == "t5" || r.spec.Architecture == "gemma3n" {
		return reference.Value{}, nil, reference.Value{}, errors.New("inference: cached layer extraction is unsupported for this architecture")
	}
	capture, err := newLayerInputCapture(layerIDs, len(r.weights.Layers))
	if err != nil {
		return reference.Value{}, nil, reference.Value{}, err
	}
	hidden, next, err := r.forwardCachedWithEmbeddingOverridesModeLocked(
		ctx, tokenIDs, cache, nil, nil, nil, nil, true, capture,
	)
	if err != nil {
		return reference.Value{}, nil, reference.Value{}, err
	}
	extracted, err := capture.result(r.spec.EmbeddingLength, len(tokenIDs))
	if err != nil {
		return reference.Value{}, nil, reference.Value{}, err
	}
	return hidden, next, extracted, nil
}

func newLayerInputCapture(layerIDs []int32, layers int) (*layerInputCapture, error) {
	if len(layerIDs) == 0 {
		return nil, errors.New("inference: extraction layer list is empty")
	}
	capture := &layerInputCapture{
		order:     append([]int32(nil), layerIDs...),
		requested: make(map[int32]struct{}, len(layerIDs)),
		values:    make(map[int32]reference.Value, len(layerIDs)),
	}
	for _, layer := range layerIDs {
		if layer < 0 || int(layer) >= layers {
			return nil, fmt.Errorf("inference: extraction layer %d is out of range", layer)
		}
		capture.requested[layer] = struct{}{}
	}
	return capture, nil
}

func (c *layerInputCapture) wants(layer int) bool {
	if c == nil {
		return false
	}
	_, ok := c.requested[int32(layer)]
	return ok
}

func (c *layerInputCapture) set(layer int, value reference.Value) {
	if !c.wants(layer) {
		return
	}
	c.values[int32(layer)] = value.Clone()
}

func (c *layerInputCapture) result(width uint32, tokens int) (reference.Value, error) {
	if c == nil || width == 0 || tokens <= 0 {
		return reference.Value{}, errors.New("inference: layer extraction state is invalid")
	}
	result := reference.Value{
		Shape: tensor.MustShape(uint64(width)*uint64(len(c.order)), uint64(tokens)),
		Data:  make([]float32, int(width)*len(c.order)*tokens),
	}
	for token := 0; token < tokens; token++ {
		for order, layer := range c.order {
			value, ok := c.values[layer]
			if !ok || value.Shape.Rank != 2 || value.Shape.Dims[0] != uint64(width) ||
				value.Shape.Dims[1] != uint64(tokens) || len(value.Data) != int(width)*tokens {
				return reference.Value{}, fmt.Errorf("inference: extraction layer %d was not captured", layer)
			}
			source := token * int(width)
			destination := (token*len(c.order) + order) * int(width)
			copy(result.Data[destination:destination+int(width)], value.Data[source:source+int(width)])
		}
	}
	return result, nil
}
