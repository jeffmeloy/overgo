package diffusionimage

import (
	"context"
	"errors"

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
	graph := newModelGraph()
	input := graph.builder.Input(block.name+".input", dtype.F32, tensor.MustShape(uint64(channels), uint64(tokens)))
	output := graph.transformerTokens(block, config, input, channels, tokens)
	if err := graph.err("transformer block"); err != nil {
		return transformerBlockProgram{}, err
	}
	return transformerBlockProgram{input: input, output: output, static: graph.static, channels: channels, tokens: tokens}, nil
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
