package scratchmodel

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// ForwardGraph is one token-batch scratch forward through shared tensor ops.
type ForwardGraph struct {
	Output     *tensor.Tensor
	Parameters map[string]*tensor.Tensor
	Mask       *tensor.Tensor
	positions  int
	mask       []float32
}

// CompileForwardGraph emits the corpus-derived model without a family runtime.
func (c Construction) CompileForwardGraph(tokens []int) (ForwardGraph, error) {
	if len(tokens) < 2 {
		return ForwardGraph{}, errors.New("scratch model: forward tokens absent")
	}
	positions := min(c.config.BlockSize, len(tokens)-1)
	for _, token := range tokens[:positions+1] {
		if token < 0 || token >= c.config.VocabSize {
			return ForwardGraph{}, errors.New("scratch model: forward token outside vocabulary")
		}
	}
	builder := tensor.NewBuilder()
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
	hidden := builder.MADNorm(
		builder.Add(builder.GetRows(parameters["wte"], tokenRows), builder.GetRows(parameters["wpe"], positionRows)),
		float32(c.config.Epsilon),
	)
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
	for layer := range c.config.LayerCount {
		prefix := fmt.Sprintf("l%d.", layer)
		qkv := builder.MulMat(parameters[prefix+"wqkv"], builder.MADNorm(hidden, float32(c.config.Epsilon)))
		var attention *tensor.Tensor
		temperature := builder.GetRows(parameters["lt"], []uint32{uint32(layer)})
		for head := range c.config.HeadCount {
			offset := uint64(head * c.config.HeadDim)
			q := builder.Reshape(
				builder.GroupSlice(qkv, offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			k := builder.Reshape(
				builder.GroupSlice(qkv, uint64(c.config.Embedding)+offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			v := builder.Reshape(
				builder.GroupSlice(qkv, uint64(2*c.config.Embedding)+offset, uint64(c.config.HeadDim), 1, uint64(3*c.config.Embedding)),
				uint64(c.config.HeadDim), uint64(positions),
			)
			temperatureScale := builder.Exp(builder.Scale(builder.FlatSlice(temperature, uint64(head), 1, 1), -1))
			scaledQuery := builder.Scale(builder.Multiply(q, temperatureScale), float32(1/math.Sqrt(float64(c.config.HeadDim))))
			scores := builder.Add(builder.Add(builder.MulMat(k, scaledQuery), lagBias), mask)
			context := builder.MulMat(builder.Transpose2D(v), builder.Softmax(scores))
			if attention == nil {
				attention = context
			} else {
				attention = builder.Concat(attention, context, 0)
			}
		}
		hidden = builder.Add(hidden, builder.MulMat(parameters[prefix+"wo"], attention))
		normalized := builder.MADNorm(hidden, float32(c.config.Epsilon))
		hidden = builder.Add(hidden, builder.MulMat(parameters[prefix+"w2"], builder.ReLU(builder.MulMat(parameters[prefix+"w1"], normalized))))
	}
	output := builder.MulMat(parameters["lm_head"], hidden)
	if err := builder.Err(); err != nil {
		return ForwardGraph{}, err
	}
	return ForwardGraph{Output: output, Parameters: parameters, Mask: mask, positions: positions, mask: maskValues}, nil
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
	result := make([]float32, len(c.weights))
	for index, value := range c.weights {
		result[index] = float32(value)
	}
	return result
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
