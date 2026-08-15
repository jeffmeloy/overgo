package diffusionimage

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type modelGraph struct {
	builder *tensor.Builder
	static  map[*tensor.Tensor]reference.Value
	bindErr error
}

func newModelGraph() *modelGraph {
	return &modelGraph{builder: tensor.NewBuilder(), static: make(map[*tensor.Tensor]reference.Value, 256)}
}

func (graph *modelGraph) bind(prefix, suffix string, data []float32, dimensions ...uint64) *tensor.Tensor {
	node := graph.builder.Input(prefix+suffix, dtype.F32, tensor.MustShape(dimensions...))
	value, err := reference.NewValue(node.Shape, data)
	if err != nil {
		graph.bindErr = errors.Join(graph.bindErr, err)
	} else {
		graph.static[node] = value
	}
	return node
}

func (graph *modelGraph) err(scope string) error {
	if graph.bindErr != nil {
		return fmt.Errorf("diffusionimage: bind %s: %w", scope, graph.bindErr)
	}
	if err := graph.builder.Err(); err != nil {
		return fmt.Errorf("diffusionimage: compile %s: %w", scope, err)
	}
	return nil
}

func (graph *modelGraph) resBlock(block *resBlock, config Config, input *tensor.Tensor, channels, height, width int) *tensor.Tensor {
	c, h, w := uint64(channels), uint64(height), uint64(width)
	norm := func(value *tensor.Tensor, weight, bias []float32, suffix string) *tensor.Tensor {
		flat := graph.builder.Reshape(value, c, h*w)
		return graph.builder.Reshape(graph.builder.GroupNorm(
			flat,
			graph.bind(block.name, suffix+".weight", weight, c),
			graph.bind(block.name, suffix+".bias", bias, c),
			uint32(config.Groups), float32(config.NormEps),
		), c, w, h)
	}
	conv := func(value *tensor.Tensor, weight, bias []float32, suffix string) *tensor.Tensor {
		return graph.builder.Conv2D(
			value,
			graph.bind(block.name, suffix+".weight", weight, 3, 3, c, c),
			graph.bind(block.name, suffix+".bias", bias, c),
			1, 1, 1, 1, 1, 1, false,
		)
	}
	hidden := conv(graph.builder.SiLU(norm(input, block.norm1Weight, block.norm1Bias, ".norm1")), block.conv1Weight, block.conv1Bias, ".conv1")
	branch := conv(graph.builder.SiLU(norm(hidden, block.norm2Weight, block.norm2Bias, ".norm2")), block.conv2Weight, block.conv2Bias, ".conv2")
	return graph.builder.Add(input, graph.builder.Scale(branch, block.residualScale))
}

func (graph *modelGraph) transformerTokens(block *attnBlock, config Config, input *tensor.Tensor, channels, tokens int) *tensor.Tensor {
	c, sequence := uint64(channels), uint64(tokens)
	heads, headDim := uint64(config.Heads), c/uint64(config.Heads)
	norm := func(value *tensor.Tensor, weight, bias []float32, suffix string) *tensor.Tensor {
		return graph.builder.AffineLayerNorm(
			value,
			graph.bind(block.name, suffix+".weight", weight, c),
			graph.bind(block.name, suffix+".bias", bias, c),
			float32(config.NormEps),
		)
	}
	linear := func(value *tensor.Tensor, weight, bias []float32, inputWidth, outputWidth uint64, suffix string) *tensor.Tensor {
		projected := graph.builder.MulMat(graph.bind(block.name, suffix+".weight", weight, inputWidth, outputWidth), value)
		if bias != nil {
			projected = graph.builder.Add(projected, graph.bind(block.name, suffix+".bias", bias, outputWidth))
		}
		return projected
	}
	contract := func(value *tensor.Tensor, wa, wb []float32, rank int, suffix string) *tensor.Tensor {
		r := uint64(rank)
		a := linear(value, wa, nil, c, heads*r, suffix+".a")
		b := linear(value, wb, nil, c, r*headDim, suffix+".b")
		var output *tensor.Tensor
		for factor := uint64(0); factor < r; factor++ {
			term := graph.builder.Multiply(
				graph.builder.GroupSlice(a, factor, 1, heads, r),
				graph.builder.GroupSlice(b, factor*headDim, headDim, 1, headDim),
			)
			if output == nil {
				output = term
			} else {
				output = graph.builder.Add(output, term)
			}
		}
		return graph.builder.Scale(output, 1/float32(rank))
	}
	xatglu := func(projected *tensor.Tensor, width uint64, alpha float32, suffix string) *tensor.Tensor {
		gate := graph.builder.Reshape(graph.builder.GroupSlice(projected, 0, width, 1, 2*width), width, sequence)
		value := graph.builder.Reshape(graph.builder.GroupSlice(projected, width, width, 1, 2*width), width, sequence)
		gate = graph.builder.Scale(graph.builder.Add(
			graph.builder.Atan(gate), graph.bind(block.name, suffix+".half_pi", []float32{math.Pi / 2}, 1),
		), 1/math.Pi)
		gate = graph.builder.Add(
			graph.builder.Scale(gate, 1+2*alpha),
			graph.bind(block.name, suffix+".negative_alpha", []float32{-alpha}, 1),
		)
		return graph.builder.Multiply(gate, value)
	}
	norm1 := norm(input, block.norm1Weight, block.norm1Bias, ".norm1")
	query := contract(norm1, block.attention.WAq, block.attention.WBq, config.QRank, ".query")
	key := contract(norm1, block.attention.WAk, block.attention.WBk, config.KVRank, ".key")
	value := contract(norm1, block.attention.WAv, block.attention.WBv, config.KVRank, ".value")
	positions := make([]uint32, tokens)
	for index := range positions {
		positions[index] = uint32(index)
	}
	query = graph.builder.RoPENeoXReverse(query, positions, uint32(headDim), float32(config.RopeTheta))
	key = graph.builder.RoPENeoXReverse(key, positions, uint32(headDim), float32(config.RopeTheta))
	attended := graph.builder.Reshape(
		graph.builder.Attention(query, key, value, 1/float32(math.Sqrt(float64(headDim))), false),
		heads*headDim, sequence,
	)
	projected := linear(attended, block.attention.Woproj, block.attention.Boproj, c, 2*c, ".attention.output")
	attention := xatglu(projected, c, float32(block.attention.AlphaO), ".attention.gate")
	residual := graph.builder.Add(input, graph.builder.Scale(attention, block.attentionScale))
	norm2 := norm(residual, block.norm2Weight, block.norm2Bias, ".norm2")
	mlp := linear(norm2, block.mlpProjection, nil, c, uint64(block.mlpProjected), ".mlp.projection")
	mlp = xatglu(mlp, uint64(block.mlpHidden), block.mlpAlpha, ".mlp.gate")
	mlp = linear(mlp, block.mlpOutput, nil, uint64(block.mlpHidden), c, ".mlp.output")
	return graph.builder.Add(residual, graph.builder.Scale(mlp, block.mlpScale))
}

func (graph *modelGraph) transformerImage(block *attnBlock, config Config, input *tensor.Tensor, channels, height, width int) *tensor.Tensor {
	tokens := uint64(height * width)
	hostSlab := graph.builder.Transpose2D(graph.builder.Reshape(input, uint64(channels), tokens))
	hostTokens := graph.builder.Reshape(hostSlab, uint64(channels), tokens)
	output := graph.transformerTokens(block, config, hostTokens, channels, int(tokens))
	hostView := graph.builder.Reshape(output, tokens, uint64(channels))
	return graph.builder.Reshape(graph.builder.Transpose2D(hostView), uint64(channels), uint64(width), uint64(height))
}

func (graph *modelGraph) projection(input *tensor.Tensor, binding convBinding, inputChannels, outputChannels, kernel, stride int) *tensor.Tensor {
	return graph.builder.Conv2D(
		input,
		graph.bind(binding.name, ".weight", binding.weight, uint64(kernel), uint64(kernel), uint64(inputChannels), uint64(outputChannels)),
		graph.bind(binding.name, ".bias", binding.bias, uint64(outputChannels)),
		uint32(stride), uint32(stride), 0, 0, 0, 0, false,
	)
}

func (graph *modelGraph) downsample(input *tensor.Tensor, binding convBinding, inputChannels, outputChannels int) *tensor.Tensor {
	projected := graph.projection(input, binding, inputChannels, outputChannels, 1, 1)
	weights := make([]float32, 4*outputChannels)
	for index := range weights {
		weights[index] = 0.25
	}
	return graph.builder.Conv2D(
		projected, graph.bind(binding.name, ".average_pool", weights, 2, 2, 1, uint64(outputChannels)), nil,
		2, 2, 0, 0, 0, 0, true,
	)
}

func (graph *modelGraph) upsample(input *tensor.Tensor, binding convBinding, inputChannels, outputChannels int) *tensor.Tensor {
	projected := graph.projection(input, binding, inputChannels, outputChannels, 1, 1)
	expanded := graph.builder.Concat(projected, projected, 0)
	return graph.builder.PixelShuffle2D(graph.builder.Concat(expanded, expanded, 0), 2)
}

func (graph *modelGraph) finalProjection(input *tensor.Tensor, binding convBinding, inputGeometry imageGeometry, outputChannels, patch int) *tensor.Tensor {
	expandedChannels := outputChannels * patch * patch
	weight := make([]float32, inputGeometry.channels*expandedChannels)
	for y := range patch {
		for x := range patch {
			for outputChannel := range outputChannels {
				row := (y*patch+x)*outputChannels + outputChannel
				for inputChannel := range inputGeometry.channels {
					source := ((inputChannel*outputChannels+outputChannel)*patch+y)*patch + x
					weight[row*inputGeometry.channels+inputChannel] = binding.weight[source]
				}
			}
		}
	}
	flat := graph.builder.Reshape(input, uint64(inputGeometry.channels), uint64(inputGeometry.height*inputGeometry.width))
	projected := graph.builder.MulMat(
		graph.bind(binding.name, ".pixel_projection", weight, uint64(inputGeometry.channels), uint64(expandedChannels)), flat,
	)
	projected = graph.builder.Reshape(projected, uint64(expandedChannels), uint64(inputGeometry.width), uint64(inputGeometry.height))
	output := graph.builder.PixelShuffle2D(projected, uint32(patch))
	return graph.builder.Add(output, graph.bind(binding.name, ".bias", binding.bias, uint64(outputChannels)))
}
