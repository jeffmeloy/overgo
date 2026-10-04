package scratchmodel

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// ForwardGraph is one token-batch scratch forward through shared tensor ops.
type ForwardGraph struct {
	Output     *tensor.Tensor
	Parameters map[string]*tensor.Tensor
	Mask       *tensor.Tensor
	Hidden     *tensor.Tensor
	Positions  int
	MaskValues []float32
	Embedding  *tensor.Tensor
	TokenRows  *tensor.Tensor
	Layers     []ForwardLayer
}

// ForwardHead is one attention head's retained forward tensors.
type ForwardHead struct {
	Query, ScaledQuery, Key, Value, Probability, TemperatureScale *tensor.Tensor
}

// ForwardLayer is one layer's retained forward tensors, the activations a
// backward consumes.
type ForwardLayer struct {
	Input, QKVNorm, QKV                *tensor.Tensor
	Attention, AttentionOutput         *tensor.Tensor
	MLPNorm, Preactivation, Activation *tensor.Tensor
	Heads                              []ForwardHead
}

// ValidateTokens admits a token batch the context and vocabulary hold and
// returns its prediction positions.
func (c Construction) ValidateTokens(tokens []int) (int, error) {
	if len(tokens) < 2 {
		return 0, errors.New("scratch model: forward tokens absent")
	}
	positions := len(tokens) - 1
	if positions > c.config.BlockSize {
		return 0, fmt.Errorf("scratch model: sequence has %d positions, context admits %d; refusing truncated evaluation", positions, c.config.BlockSize)
	}
	for _, token := range tokens {
		if token < 0 || token >= c.config.VocabSize {
			return 0, errors.New("scratch model: forward token outside vocabulary")
		}
	}
	return positions, nil
}

// CompileForwardGraph emits the corpus-derived model without a family runtime.
func (c Construction) CompileForwardGraph(tokens []int) (ForwardGraph, error) {
	positions, err := c.ValidateTokens(tokens)
	if err != nil {
		return ForwardGraph{}, err
	}
	// The registered architecture profile drives the constructed forward:
	// normalization and feed-forward come from the executor's policy
	// vocabulary, so a policy the vocabulary cannot express cannot compile.
	profile := c.Architecture()
	if profile.Normalization != model.NormalizationMAD ||
		profile.FeedForward != model.FeedForwardReLU ||
		profile.Position != model.PositionLearnedAbsolute {
		return ForwardGraph{}, fmt.Errorf("scratch model: architecture %q policies do not describe the constructed topology", profile.Name)
	}
	builder := tensor.NewBuilder()
	norm := func(value *tensor.Tensor) *tensor.Tensor {
		return builder.MADNorm(value, float32(c.config.Epsilon))
	}
	activate := builder.ReLU
	parameters := make(map[string]*tensor.Tensor, len(c.parameters))
	for _, parameter := range c.parameters {
		parameters[parameter.Name] = builder.Input(
			parameter.Name,
			dtype.F32,
			tensor.MustShape(uint64(parameter.Cols), uint64(parameter.Rows)),
		)
	}
	tokenRows, positionRows := make([]uint32, positions), make([]uint32, positions)
	for position := range positions {
		tokenRows[position], positionRows[position] = uint32(tokens[position]), uint32(position)
	}
	tokenLookup := builder.GetRows(parameters["wte"], tokenRows)
	embedding := builder.Add(tokenLookup, builder.GetRows(parameters["wpe"], positionRows))
	hidden := norm(embedding)
	maskValues := causalWindowMask(positions, c.config.AttentionWindow)
	mask := builder.Input("causal-window-mask", dtype.F32, tensor.MustShape(uint64(positions), uint64(positions)))
	lagRows := make([]uint32, positions*positions)
	for query := range positions {
		for key := range positions {
			if key <= query {
				lagRows[query*positions+key] = uint32(min(c.config.BlockSize-1, query-key))
			}
		}
	}
	positionBias := builder.Reshape(parameters["pos_bias"], 1, uint64(c.config.BlockSize))
	lagBias := builder.Reshape(builder.GetRows(positionBias, lagRows), uint64(positions), uint64(positions))
	layers := make([]ForwardLayer, c.config.LayerCount)
	for layer := range c.config.LayerCount {
		prefix := fmt.Sprintf("l%d.", layer)
		cache := ForwardLayer{Input: hidden, Heads: make([]ForwardHead, c.config.HeadCount)}
		cache.QKVNorm = norm(hidden)
		cache.QKV = builder.MulMat(parameters[prefix+"wqkv"], cache.QKVNorm)
		var attention *tensor.Tensor
		temperature := builder.GetRows(parameters["lt"], []uint32{uint32(layer)})
		for head := range c.config.HeadCount {
			offset := uint64(head * c.config.HeadDim)
			q := builder.Reshape(
				builder.GroupSlice(cache.QKV, offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			k := builder.Reshape(
				builder.GroupSlice(cache.QKV, uint64(c.config.Embedding)+offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			v := builder.Reshape(
				builder.GroupSlice(cache.QKV, uint64(2*c.config.Embedding)+offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			temperatureScale := builder.Exp(builder.Scale(builder.FlatSlice(temperature, uint64(head), 1, 1), -1))
			scaledQuery := builder.Scale(builder.Multiply(q, temperatureScale), float32(1/math.Sqrt(float64(c.config.HeadDim))))
			scores := builder.Add(builder.Add(builder.MulMat(k, scaledQuery), lagBias), mask)
			probability := builder.Softmax(scores)
			context := builder.MulMat(builder.Transpose2D(v), probability)
			cache.Heads[head] = ForwardHead{
				Query: q, ScaledQuery: scaledQuery, Key: k, Value: v,
				Probability: probability, TemperatureScale: temperatureScale,
			}
			if attention == nil {
				attention = context
			} else {
				attention = builder.Concat(attention, context, 0)
			}
		}
		cache.Attention = attention
		cache.AttentionOutput = builder.Add(hidden, builder.MulMat(parameters[prefix+"wo"], attention))
		cache.MLPNorm = norm(cache.AttentionOutput)
		cache.Preactivation = builder.MulMat(parameters[prefix+"w1"], cache.MLPNorm)
		cache.Activation = activate(cache.Preactivation)
		hidden = builder.Add(cache.AttentionOutput, builder.MulMat(parameters[prefix+"w2"], cache.Activation))
		layers[layer] = cache
	}
	output := builder.MulMat(parameters["lm_head"], hidden)
	if err := builder.Err(); err != nil {
		return ForwardGraph{}, err
	}
	return ForwardGraph{
		Output: output, Parameters: parameters, Mask: mask, Hidden: hidden,
		Positions: positions, MaskValues: maskValues, Embedding: embedding, TokenRows: tokenLookup, Layers: layers,
	}, nil
}

// CacheOutputs lists the forward tensors a backward reads: the output, the
// hidden and embedded streams, and every layer's and head's activations.
func (g ForwardGraph) CacheOutputs() []*tensor.Tensor {
	outputs := []*tensor.Tensor{g.Output, g.Hidden, g.Embedding}
	for _, layer := range g.Layers {
		outputs = append(outputs, layer.Input, layer.QKVNorm, layer.QKV, layer.Attention, layer.AttentionOutput, layer.MLPNorm, layer.Preactivation, layer.Activation)
		for _, head := range layer.Heads {
			outputs = append(outputs, head.Query, head.ScaledQuery, head.Key, head.Value, head.Probability, head.TemperatureScale)
		}
	}
	return outputs
}

// Feeds materializes graph feeds from one flat parameter slab.
func (g ForwardGraph) Feeds(c Construction, slab []float32) (map[*tensor.Tensor]reference.Value, error) {
	if g.Output == nil || g.Mask == nil || g.Positions <= 0 || len(g.Parameters) != len(c.parameters) || len(slab) != len(c.weights) {
		return nil, errors.New("scratch model: forward graph or slab differs")
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(g.Parameters)+1)
	for name, input := range g.Parameters {
		binding, ok := c.bindings[name]
		if !ok {
			return nil, fmt.Errorf("scratch model: graph parameter %q has no slab binding", name)
		}
		value, err := reference.NewValue(input.Shape, slab[binding.start:binding.end])
		if err != nil {
			return nil, err
		}
		feeds[input] = value
	}
	mask, err := reference.NewValue(g.Mask.Shape, g.MaskValues)
	if err != nil {
		return nil, err
	}
	feeds[g.Mask] = mask
	return feeds, nil
}

// InitialWeights copies the constructed parameter slab, each parameter at
// its Offset.
func (c Construction) InitialWeights() []float32 {
	return slices.Clone(c.weights)
}

func causalWindowMask(positions, window int) []float32 {
	result := make([]float32, positions*positions)
	for query := range positions {
		start := max(0, query+1-window)
		for key := range positions {
			if key < start || key > query {
				result[query*positions+key] = -math.MaxFloat32
			}
		}
	}
	return result
}
