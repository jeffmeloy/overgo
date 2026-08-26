package diffusionimage

import (
	"context"
	"errors"
	"fmt"
	"math/rand"

	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type ResidentForward struct {
	session *graphruntime.ResidentSession
	program *graphruntime.ResidentProgram
	input   *tensor.Tensor
	output  *tensor.Tensor
	image   imageGeometry
}

func CompileResidentForward(ctx context.Context, model *Model, ordinal, imageHeight, imageWidth int) (*ResidentForward, error) {
	if model == nil || imageHeight <= 0 || imageWidth <= 0 || imageHeight%model.Cfg.PatchSize != 0 || imageWidth%model.Cfg.PatchSize != 0 {
		return nil, errors.New("diffusionimage: invalid resident forward request")
	}
	graph := newModelGraph()
	input := graph.builder.Input("simplediffusion.input", dtype.F32, tensor.MustShape(
		uint64(model.Cfg.InChannels), uint64(imageWidth), uint64(imageHeight),
	))
	cur := graph.projection(input, model.patch, model.Cfg.InChannels, model.Cfg.BaseChannels, model.Cfg.PatchSize, model.Cfg.PatchSize)
	channels := model.Cfg.BaseChannels
	height, width := imageHeight/model.Cfg.PatchSize, imageWidth/model.Cfg.PatchSize
	currResidual := cur
	residuals := make([]*tensor.Tensor, 0, len(model.encoders)-1)
	applyBlock := func(owner block, value *tensor.Tensor) *tensor.Tensor {
		switch typed := owner.(type) {
		case *resBlock:
			return graph.resBlock(typed, model.Cfg, value, channels, height, width)
		case *attnBlock:
			return graph.transformerImage(typed, model.Cfg, value, channels, height, width)
		default:
			graph.bindErr = errors.Join(graph.bindErr, fmt.Errorf("unsupported block %T", owner))
			return value
		}
	}
	for levelIndex := range model.encoders {
		level := &model.encoders[levelIndex]
		for _, owner := range level.blocks {
			cur = applyBlock(owner, cur)
		}
		if levelIndex < len(model.encoders)-1 {
			residuals = append(residuals, currResidual)
			cur = graph.downsample(cur, level.transition, channels, channels*2)
			channels, height, width = channels*2, height/2, width/2
			currResidual = cur
		}
	}
	for _, owner := range model.middle {
		cur = applyBlock(owner, cur)
	}
	cur = graph.builder.Add(cur, graph.builder.Scale(currResidual, -model.middleResidual))
	for levelIndex := range model.decoders {
		level := &model.decoders[levelIndex]
		for _, owner := range level.blocks {
			cur = applyBlock(owner, cur)
		}
		if levelIndex < len(model.decoders)-1 {
			cur = graph.upsample(cur, level.transition, channels, channels/2)
			channels, height, width = channels/2, height*2, width*2
			residual := residuals[len(residuals)-1]
			residuals = residuals[:len(residuals)-1]
			cur = graph.builder.Add(cur, graph.builder.Scale(residual, model.middleResidual))
		}
	}
	output := graph.finalProjection(cur, model.final, imageGeometry{
		channels: channels, height: height, width: width,
	}, model.Cfg.InChannels, model.Cfg.PatchSize)
	if err := graph.err("resident forward"); err != nil {
		return nil, err
	}
	session, err := graphruntime.NewResidentSession(ordinal)
	if err != nil {
		return nil, err
	}
	program, err := session.Compile(ctx, "diffusion image resident forward", []*tensor.Tensor{input}, output)
	if err == nil {
		for _, binding := range graph.static {
			err = session.BindF32(ctx, program, "diffusionimage:"+binding.node.Name, binding.node, binding.value.Data)
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		return nil, errors.Join(err, session.Close(context.Background()))
	}
	session.Seal()
	return &ResidentForward{
		session: session, program: program, input: input, output: output,
		image: imageGeometry{channels: model.Cfg.InChannels, height: imageHeight, width: imageWidth},
	}, nil
}

func (forward *ResidentForward) Execute(ctx context.Context, input []float32) ([]float32, error) {
	if forward == nil || forward.session == nil || len(input) != forward.image.channels*forward.image.height*forward.image.width {
		return nil, errors.New("diffusionimage: resident forward input mismatch")
	}
	graphInput := nchwToGraphImage(input, forward.image.channels, forward.image.height, forward.image.width)
	value, err := reference.NewValue(forward.input.Shape, graphInput)
	if err != nil {
		return nil, err
	}
	result, err := forward.session.Execute(ctx, forward.program, map[*tensor.Tensor]reference.Value{forward.input: value})
	if err != nil {
		return nil, err
	}
	return graphImageToNCHW(result[forward.output].Data, forward.image.channels, forward.image.height, forward.image.width), nil
}

func (forward *ResidentForward) Sample(ctx context.Context, steps int, seed int64) ([]float32, error) {
	if forward == nil || forward.session == nil || steps <= 0 {
		return nil, errors.New("diffusionimage: resident sample request is invalid")
	}
	rng := rand.New(rand.NewSource(seed))
	pixels := make([]float32, forward.image.channels*forward.image.height*forward.image.width)
	for index := range pixels {
		pixels[index] = float32(rng.NormFloat64())
	}
	delta := float32(1) / float32(steps)
	for range steps {
		velocity, err := forward.Execute(ctx, pixels)
		if err != nil {
			return nil, err
		}
		for index := range pixels {
			pixels[index] += velocity[index] * delta
		}
	}
	return pixels, nil
}

func (forward *ResidentForward) Stats(ctx context.Context) (graphruntime.ResidentStats, error) {
	if forward == nil || forward.session == nil {
		return graphruntime.ResidentStats{}, errors.New("diffusionimage: resident forward is unavailable")
	}
	return forward.session.Stats(ctx)
}

func (forward *ResidentForward) Close(ctx context.Context) error {
	if forward == nil || forward.session == nil {
		return nil
	}
	err := forward.session.Close(ctx)
	forward.session = nil
	forward.program = nil
	return err
}
