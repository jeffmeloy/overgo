package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type layerInputCapture struct {
	order     []int32
	requested map[int32]struct{}
	values    map[int32]reference.Value
	// Optional exact-attention inputs.
	attnLayer int32
	attnScale float32
	attnHeads int
	attnKV    int
	attnDim   int
	attnQuery reference.Value
	attnKey   reference.Value
}

// AttentionCapture holds one layer's exact host-replay inputs.
type AttentionCapture struct {
	Layer   int
	Tokens  int
	Heads   int
	KVHeads int
	HeadDim int
	Scale   float32
	Query   reference.Value
	Key     reference.Value
}

// AttentionCaptureLayers reports exactly replayable layers.
func (r *Runner) AttentionCaptureLayers() []int32 {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || !r.forwardProgram().LayerCapture() {
		return nil
	}
	result := make([]int32, 0, len(r.weights.Layers))
	for layer := range r.weights.Layers {
		if exactAttentionCapture(r.layerProgram(layer).Layer()) {
			result = append(result, int32(layer))
		}
	}
	return result
}

func exactAttentionCapture(plan model.LayerPlan) bool {
	attention := plan.AttentionGraph
	return plan.HasKV && attention.Causal && !attention.UseSinks && !attention.ChunkedWindow &&
		attention.Window == 0 && attention.Softcap == 0 && attention.MaxALiBiBias == 0
}

// ExtractAttention records one exact-replay query/key boundary.
func (r *Runner) ExtractAttention(ctx context.Context, tokenIDs []tokenizer.TokenID, layer int32) (AttentionCapture, error) {
	if r == nil {
		return AttentionCapture{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return AttentionCapture{}, errors.New("inference: runner is closed")
	}
	if !r.forwardProgram().LayerCapture() {
		return AttentionCapture{}, errors.New("inference: attention capture is unsupported for this architecture")
	}
	if layer < 0 || int(layer) >= len(r.weights.Layers) {
		return AttentionCapture{}, fmt.Errorf("inference: attention layer %d is out of range", layer)
	}
	if !exactAttentionCapture(r.layerProgram(int(layer)).Layer()) {
		return AttentionCapture{}, fmt.Errorf("inference: attention layer %d policy cannot be replayed exactly", layer)
	}
	capture := &layerInputCapture{
		requested: map[int32]struct{}{},
		values:    map[int32]reference.Value{},
		attnLayer: layer,
	}
	if _, _, err := r.forwardCachedProjectedChunkModeLocked(
		ctx, tokenIDs, nil, ProjectedInputs{}, true, capture,
	); err != nil {
		return AttentionCapture{}, err
	}
	if capture.attnQuery.Data == nil || capture.attnKey.Data == nil {
		return AttentionCapture{}, fmt.Errorf("inference: layer %d does not expose attention query (unsupported block type)", layer)
	}
	return AttentionCapture{
		Layer: int(layer), Tokens: len(tokenIDs),
		Heads: capture.attnHeads, KVHeads: capture.attnKV, HeadDim: capture.attnDim,
		Scale: capture.attnScale, Query: capture.attnQuery, Key: capture.attnKey,
	}, nil
}

// ExtractLayerInputs: full-sequence pre-layer hidden rows.
func (r *Runner) ExtractLayerInputs(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	layerIDs []int32,
) (reference.Value, error) {
	_, _, features, err := r.ForwardCachedExtractLayerInputs(ctx, tokenIDs, nil, layerIDs)
	return features, err
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
	if !r.forwardProgram().LayerCapture() {
		return reference.Value{}, nil, reference.Value{}, errors.New("inference: cached layer extraction is unsupported for this architecture")
	}
	capture, err := newLayerInputCapture(layerIDs, len(r.weights.Layers))
	if err != nil {
		return reference.Value{}, nil, reference.Value{}, err
	}
	hidden, next, err := r.forwardCachedProjectedChunkModeLocked(
		ctx, tokenIDs, cache, ProjectedInputs{}, true, capture,
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
		order:     slices.Clone(layerIDs),
		requested: make(map[int32]struct{}, len(layerIDs)),
		values:    make(map[int32]reference.Value, len(layerIDs)),
		attnLayer: -1,
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
