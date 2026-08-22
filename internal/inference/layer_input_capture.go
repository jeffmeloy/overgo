package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type layerInputCapture struct {
	order     []int32
	requested map[int32]struct{}
	values    map[int32]reference.Value
	// Optional exact-attention inputs.
	attention bool
	attnLayer int32
	attnScale float32
	attnHeads int
	attnKV    int
	attnDim   int
	attnQuery reference.Value
	attnKey   reference.Value
}

// AttentionCapture: exact host-replay inputs.
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

// AttentionCaptureLayers returns replayable layers.
func (r *Runner) AttentionCaptureLayers() []int32 {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || !r.forwardProgram().LayerCapture() {
		return nil
	}
	var result []int32
	for layer := range r.weights.Layers {
		if r.layerProgram(layer).Layer().ExactAttentionReplay() {
			result = append(result, int32(layer))
		}
	}
	return result
}

// ExtractAttention captures one replay boundary.
func (r *Runner) ExtractAttention(ctx context.Context, tokenIDs []tokenizer.TokenID, layer int32) (AttentionCapture, error) {
	if r == nil {
		return AttentionCapture{}, errRunnerNil
	}
	if err := r.lockOpen(); err != nil {
		return AttentionCapture{}, err
	}
	defer r.mu.Unlock()
	if !r.forwardProgram().LayerCapture() {
		return AttentionCapture{}, errors.New("inference: attention capture is unsupported for this architecture")
	}
	if !checked.NonNegativeInts(int(layer)) || int(layer) >= len(r.weights.Layers) {
		return AttentionCapture{}, fmt.Errorf("inference: attention layer %d is out of range", layer)
	}
	if !r.layerProgram(int(layer)).Layer().ExactAttentionReplay() {
		return AttentionCapture{}, fmt.Errorf("inference: attention layer %d policy cannot be replayed exactly", layer)
	}
	capture := &layerInputCapture{
		requested: map[int32]struct{}{},
		attention: true, attnLayer: layer,
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
		return reference.Value{}, nil, reference.Value{}, errRunnerNil
	}
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, nil, reference.Value{}, err
	}
	defer r.mu.Unlock()
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
	extracted, err := capture.result(int(r.spec.EmbeddingLength), len(tokenIDs))
	if err != nil {
		return reference.Value{}, nil, reference.Value{}, err
	}
	return hidden, next, extracted, nil
}

func newLayerInputCapture(layerIDs []int32, layers int) (*layerInputCapture, error) {
	if !checked.Nonzero(len(layerIDs)) {
		return nil, errors.New("inference: extraction layer list is empty")
	}
	capture := &layerInputCapture{
		order:     slices.Clone(layerIDs),
		requested: make(map[int32]struct{}, len(layerIDs)),
		values:    make(map[int32]reference.Value, len(layerIDs)),
	}
	for _, layer := range layerIDs {
		if !checked.NonNegativeInts(int(layer)) || int(layer) >= layers {
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
	c.values[int32(layer)] = value
}

func (c *layerInputCapture) result(expectedExtent, expectedTokens int) (reference.Value, error) {
	if c == nil || !checked.PositiveInts(expectedExtent, expectedTokens) {
		return reference.Value{}, errors.New("inference: layer extraction state is invalid")
	}
	result := reference.Value{
		Shape: tensor.MustShape(uint64(expectedExtent*len(c.order)), uint64(expectedTokens)),
		Data:  make([]float32, expectedExtent*len(c.order)*expectedTokens),
	}
	for order, layer := range c.order {
		value, ok := c.values[layer]
		extent, tokens, valid := value.MatrixExtents()
		if !ok || !valid || extent != expectedExtent || tokens != expectedTokens {
			return reference.Value{}, fmt.Errorf("inference: extraction layer %d was not captured", layer)
		}
		for token := range expectedTokens {
			source := token * expectedExtent
			destination := (token*len(c.order) + order) * expectedExtent
			copy(result.Data[destination:destination+expectedExtent], value.Data[source:source+expectedExtent])
		}
	}
	return result, nil
}
