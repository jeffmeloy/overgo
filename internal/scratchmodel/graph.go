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
	positions  int
	mask       []float32
	embedding  *tensor.Tensor
	tokenRows  *tensor.Tensor
	layers     []forwardLayer
}

type forwardHead struct {
	query, scaledQuery, key, value, probability, temperatureScale *tensor.Tensor
}

type forwardLayer struct {
	input, qkvNorm, qkv                *tensor.Tensor
	attention, attentionOutput         *tensor.Tensor
	mlpNorm, preactivation, activation *tensor.Tensor
	heads                              []forwardHead
}

func (c Construction) validateTokens(tokens []int) (int, error) {
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
	positions, err := c.validateTokens(tokens)
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
	layers := make([]forwardLayer, c.config.LayerCount)
	for layer := range c.config.LayerCount {
		prefix := fmt.Sprintf("l%d.", layer)
		cache := forwardLayer{input: hidden, heads: make([]forwardHead, c.config.HeadCount)}
		cache.qkvNorm = norm(hidden)
		cache.qkv = builder.MulMat(parameters[prefix+"wqkv"], cache.qkvNorm)
		var attention *tensor.Tensor
		temperature := builder.GetRows(parameters["lt"], []uint32{uint32(layer)})
		for head := range c.config.HeadCount {
			offset := uint64(head * c.config.HeadDim)
			q := builder.Reshape(
				builder.GroupSlice(cache.qkv, offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			k := builder.Reshape(
				builder.GroupSlice(cache.qkv, uint64(c.config.Embedding)+offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			v := builder.Reshape(
				builder.GroupSlice(cache.qkv, uint64(2*c.config.Embedding)+offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			temperatureScale := builder.Exp(builder.Scale(builder.FlatSlice(temperature, uint64(head), 1, 1), -1))
			scaledQuery := builder.Scale(builder.Multiply(q, temperatureScale), float32(1/math.Sqrt(float64(c.config.HeadDim))))
			scores := builder.Add(builder.Add(builder.MulMat(k, scaledQuery), lagBias), mask)
			probability := builder.Softmax(scores)
			context := builder.MulMat(builder.Transpose2D(v), probability)
			cache.heads[head] = forwardHead{
				query: q, scaledQuery: scaledQuery, key: k, value: v,
				probability: probability, temperatureScale: temperatureScale,
			}
			if attention == nil {
				attention = context
			} else {
				attention = builder.Concat(attention, context, 0)
			}
		}
		cache.attention = attention
		cache.attentionOutput = builder.Add(hidden, builder.MulMat(parameters[prefix+"wo"], attention))
		cache.mlpNorm = norm(cache.attentionOutput)
		cache.preactivation = builder.MulMat(parameters[prefix+"w1"], cache.mlpNorm)
		cache.activation = activate(cache.preactivation)
		hidden = builder.Add(cache.attentionOutput, builder.MulMat(parameters[prefix+"w2"], cache.activation))
		layers[layer] = cache
	}
	output := builder.MulMat(parameters["lm_head"], hidden)
	if err := builder.Err(); err != nil {
		return ForwardGraph{}, err
	}
	return ForwardGraph{
		Output: output, Parameters: parameters, Mask: mask, Hidden: hidden,
		positions: positions, mask: maskValues, embedding: embedding, tokenRows: tokenLookup, layers: layers,
	}, nil
}

func (g ForwardGraph) cacheOutputs() []*tensor.Tensor {
	outputs := []*tensor.Tensor{g.Output, g.Hidden, g.embedding}
	for _, layer := range g.layers {
		outputs = append(outputs, layer.input, layer.qkvNorm, layer.qkv, layer.attention, layer.attentionOutput, layer.mlpNorm, layer.preactivation, layer.activation)
		for _, head := range layer.heads {
			outputs = append(outputs, head.query, head.scaledQuery, head.key, head.value, head.probability, head.temperatureScale)
		}
	}
	return outputs
}

// Feeds materializes graph feeds from one flat parameter slab.
func (g ForwardGraph) Feeds(c Construction, slab []float32) (map[*tensor.Tensor]reference.Value, error) {
	if g.Output == nil || g.Mask == nil || g.positions <= 0 || len(g.Parameters) != len(c.parameters) || len(slab) != len(c.weights) {
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
	mask, err := reference.NewValue(g.Mask.Shape, g.mask)
	if err != nil {
		return nil, err
	}
	feeds[g.Mask] = mask
	return feeds, nil
}

func (c Construction) initialF32() []float32 {
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
