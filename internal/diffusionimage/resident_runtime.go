package diffusionimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/latentimage"
)

type ResidentGenerator struct {
	model   *Model
	forward *ResidentForward
	height  int
	width   int
}

// SessionKey identifies reusable resident state for the request.
func (request Request) SessionKey() (string, error) {
	return fmt.Sprintf("f32:%dx%d", request.Height, request.Width), nil
}

func LoadResidentGenerator(ctx context.Context, path string, request Request) (*ResidentGenerator, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := ValidateRequest(request); err != nil {
		return nil, err
	}
	model, err := Load(path)
	if err != nil {
		return nil, err
	}
	if _, err := model.prepare(ctx, request); err != nil {
		return nil, err
	}
	forward, err := CompileResidentForward(ctx, model, 0, request.Height, request.Width)
	if err != nil {
		return nil, err
	}
	return &ResidentGenerator{model: model, forward: forward, height: request.Height, width: request.Width}, nil
}

func (generator *ResidentGenerator) Reset(ctx context.Context, request Request) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if generator == nil || generator.forward == nil {
		return errors.New("diffusionimage: resident generator is unavailable")
	}
	if err := ValidateRequest(request); err != nil {
		return err
	}
	if request.Height != generator.height || request.Width != generator.width {
		return errors.New("diffusionimage: resident generator geometry differs")
	}
	return nil
}

func (generator *ResidentGenerator) prepare(ctx context.Context, request Request) (samplePlan, error) {
	if err := generator.Reset(ctx, request); err != nil {
		return samplePlan{}, err
	}
	return generator.model.prepare(ctx, request)
}

func (generator *ResidentGenerator) integrate(ctx context.Context, plan samplePlan) (sampleFeatures, error) {
	request := plan.request
	pixels, err := generator.forward.Sample(ctx, request.Steps, request.Seed)
	return sampleFeatures{pixels: pixels, height: request.Height, width: request.Width}, err
}

func (generator *ResidentGenerator) decode(ctx context.Context, features sampleFeatures) (latentimage.EncodedImage, error) {
	return generator.model.decode(ctx, features)
}

func (generator *ResidentGenerator) Close(ctx context.Context) error {
	if generator == nil || generator.forward == nil {
		return nil
	}
	err := generator.forward.Close(context.WithoutCancel(ctx))
	generator.forward = nil
	generator.model = nil
	return err
}
