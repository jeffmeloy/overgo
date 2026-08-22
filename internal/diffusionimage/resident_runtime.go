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
	if err := ValidateRequest(request); err != nil {
		return nil, err
	}
	model, err := Load(path)
	if err != nil {
		return nil, err
	}
	if _, err := model.prepare(request); err != nil {
		return nil, err
	}
	forward, err := CompileResidentForward(ctx, model, 0, request.Height, request.Width)
	if err != nil {
		return nil, err
	}
	return &ResidentGenerator{model: model, forward: forward, height: request.Height, width: request.Width}, nil
}

func (generator *ResidentGenerator) Reset(_ context.Context, request Request) error {
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

func (generator *ResidentGenerator) prepare(request Request) (samplePlan, error) {
	if err := generator.Reset(context.Background(), request); err != nil {
		return samplePlan{}, err
	}
	return generator.model.prepare(request)
}

func (generator *ResidentGenerator) integrate(plan samplePlan) (sampleFeatures, error) {
	request := plan.request
	pixels, err := generator.forward.Sample(context.Background(), request.Steps, request.Seed)
	return sampleFeatures{pixels: pixels, height: request.Height, width: request.Width}, err
}

func (generator *ResidentGenerator) decode(features sampleFeatures) (latentimage.EncodedImage, error) {
	return generator.model.decode(features)
}

func (generator *ResidentGenerator) Close(ctx context.Context) error {
	if generator == nil || generator.forward == nil {
		return nil
	}
	err := generator.forward.Close(ctx)
	generator.forward = nil
	generator.model = nil
	return err
}
