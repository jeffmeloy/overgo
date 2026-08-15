package diffusionimage

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/executor"
	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type transformerBlockProgram struct {
	input, output *tensor.Tensor
	static        map[*tensor.Tensor]reference.Value
	channels      int
	tokens        int
}

func compileTransformerBlockProgram(block *attnBlock, config Config, channels, tokens int) (transformerBlockProgram, error) {
	if block == nil || channels <= 0 || tokens <= 0 || config.Heads <= 0 || channels%config.Heads != 0 ||
		config.QRank <= 0 || config.KVRank <= 0 || config.RopeTheta <= 0 || config.NormEps <= 0 {
		return transformerBlockProgram{}, errors.New("diffusionimage: invalid transformer block program")
	}
	builder := tensor.NewBuilder()
	static := make(map[*tensor.Tensor]reference.Value, 20)
	var bindErr error
	bind := func(name string, data []float32, dimensions ...uint64) *tensor.Tensor {
		node := builder.Input(block.name+name, dtype.F32, tensor.MustShape(dimensions...))
		value, err := reference.NewValue(node.Shape, data)
		if err != nil {
			bindErr = errors.Join(bindErr, err)
		} else {
			static[node] = value
		}
		return node
	}
	c, sequence := uint64(channels), uint64(tokens)
	heads, headDim := uint64(config.Heads), c/uint64(config.Heads)
	input := builder.Input(block.name+".input", dtype.F32, tensor.MustShape(c, sequence))
	norm := func(value *tensor.Tensor, weight, bias []float32, suffix string) *tensor.Tensor {
		return builder.AffineLayerNorm(
			value,
			bind(suffix+".weight", weight, c),
			bind(suffix+".bias", bias, c),
			float32(config.NormEps),
		)
	}
	linear := func(value *tensor.Tensor, weight, bias []float32, inputWidth, outputWidth uint64, suffix string) *tensor.Tensor {
		projected := builder.MulMat(bind(suffix+".weight", weight, inputWidth, outputWidth), value)
		if bias != nil {
			projected = builder.Add(projected, bind(suffix+".bias", bias, outputWidth))
		}
		return projected
	}
	contract := func(value *tensor.Tensor, wa, wb []float32, rank int, suffix string) *tensor.Tensor {
		r := uint64(rank)
		a := linear(value, wa, nil, c, heads*r, suffix+".a")
		b := linear(value, wb, nil, c, r*headDim, suffix+".b")
		var output *tensor.Tensor
		for factor := uint64(0); factor < r; factor++ {
			aFactor := builder.GroupSlice(a, factor, 1, heads, r)
			bFactor := builder.GroupSlice(b, factor*headDim, headDim, 1, headDim)
			term := builder.Multiply(aFactor, bFactor)
			if output == nil {
				output = term
			} else {
				output = builder.Add(output, term)
			}
		}
		return builder.Scale(output, 1/float32(rank))
	}
	xatglu := func(projected *tensor.Tensor, width uint64, alpha float32, suffix string) *tensor.Tensor {
		gate := builder.Reshape(builder.GroupSlice(projected, 0, width, 1, 2*width), width, sequence)
		value := builder.Reshape(builder.GroupSlice(projected, width, width, 1, 2*width), width, sequence)
		gate = builder.Scale(builder.Add(builder.Atan(gate), bind(suffix+".half_pi", []float32{math.Pi / 2}, 1)), 1/math.Pi)
		gate = builder.Add(builder.Scale(gate, 1+2*alpha), bind(suffix+".negative_alpha", []float32{-alpha}, 1))
		return builder.Multiply(gate, value)
	}

	norm1 := norm(input, block.norm1Weight, block.norm1Bias, ".norm1")
	query := contract(norm1, block.attention.WAq, block.attention.WBq, config.QRank, ".query")
	key := contract(norm1, block.attention.WAk, block.attention.WBk, config.KVRank, ".key")
	value := contract(norm1, block.attention.WAv, block.attention.WBv, config.KVRank, ".value")
	positions := make([]uint32, tokens)
	for index := range positions {
		positions[index] = uint32(index)
	}
	query = builder.RoPENeoXReverse(query, positions, uint32(headDim), float32(config.RopeTheta))
	key = builder.RoPENeoXReverse(key, positions, uint32(headDim), float32(config.RopeTheta))
	attended := builder.Reshape(
		builder.Attention(query, key, value, 1/float32(math.Sqrt(float64(headDim))), false),
		heads*headDim, sequence,
	)
	attentionProjection := linear(attended, block.attention.Woproj, block.attention.Boproj, c, 2*c, ".attention.output")
	attention := xatglu(attentionProjection, c, float32(block.attention.AlphaO), ".attention.gate")
	residual := builder.Add(input, builder.Scale(attention, block.attentionScale))
	norm2 := norm(residual, block.norm2Weight, block.norm2Bias, ".norm2")
	mlpProjection := linear(norm2, block.mlpProjection, nil, c, uint64(block.mlpProjected), ".mlp.projection")
	mlpHidden := xatglu(mlpProjection, uint64(block.mlpHidden), block.mlpAlpha, ".mlp.gate")
	mlpOutput := linear(mlpHidden, block.mlpOutput, nil, uint64(block.mlpHidden), c, ".mlp.output")
	output := builder.Add(residual, builder.Scale(mlpOutput, block.mlpScale))
	if bindErr != nil {
		return transformerBlockProgram{}, fmt.Errorf("diffusionimage: bind transformer block: %w", bindErr)
	}
	if err := builder.Err(); err != nil {
		return transformerBlockProgram{}, fmt.Errorf("diffusionimage: compile transformer block: %w", err)
	}
	return transformerBlockProgram{input: input, output: output, static: static, channels: channels, tokens: tokens}, nil
}

func (program transformerBlockProgram) execute(ctx context.Context, device *executor.Executor, input []float32) ([]float32, error) {
	if program.input == nil || program.output == nil || len(input) != program.channels*program.tokens {
		return nil, errors.New("diffusionimage: transformer block input mismatch")
	}
	value, err := reference.NewValue(program.input.Shape, input)
	if err != nil {
		return nil, err
	}
	feeds := graphruntime.NewFeeds()
	feeds.AddHost(program.static)
	feeds.Host[program.input] = value
	result, err := feeds.Execute(ctx, []*tensor.Tensor{program.output}, device)
	if err != nil {
		return nil, err
	}
	return result[program.output].Data, nil
}
