package diffusionimage

import (
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
)

type generatorFunc struct {
	prepareFunc   func(Request) (samplePlan, error)
	integrateFunc func(samplePlan) (sampleFeatures, error)
	decodeFunc    func(sampleFeatures) (latentimage.EncodedImage, error)
}

func (f generatorFunc) prepare(request Request) (samplePlan, error) { return f.prepareFunc(request) }
func (f generatorFunc) integrate(plan samplePlan) (sampleFeatures, error) {
	return f.integrateFunc(plan)
}
func (f generatorFunc) decode(features sampleFeatures) (latentimage.EncodedImage, error) {
	return f.decodeFunc(features)
}

func TestRegisteredRuntimeExecutesImageProgram(t *testing.T) {
	fixture := modelrecipetest.NewDefinedCapability(t, "diffusion-image", modelrecipe.DiffusionImageDefinition)
	request := Request{Seed: 7, Steps: 2, Height: 64, Width: 64}
	want := latentimage.EncodedImage{Data: []byte{1, 2, 3}, MediaType: "image/png", Channels: 3, Height: 64, Width: 64}
	stages := generatorFunc{
		prepareFunc: func(got Request) (samplePlan, error) {
			if got != request {
				t.Fatalf("request = %+v", got)
			}
			return samplePlan{request: got}, nil
		},
		integrateFunc: func(plan samplePlan) (sampleFeatures, error) {
			return sampleFeatures{pixels: []float32{0.25}, height: plan.request.Height, width: plan.request.Width}, nil
		},
		decodeFunc: func(features sampleFeatures) (latentimage.EncodedImage, error) {
			if len(features.pixels) != 1 || features.height != 64 || features.width != 64 {
				t.Fatalf("features = %+v", features)
			}
			return want, nil
		},
	}
	if err := registerRuntime(fixture.Runtime, fixture.Model, stages); err != nil {
		t.Fatal(err)
	}
	got := modelrecipetest.MustExecuteScalar[latentimage.EncodedImage](t, fixture, "diffusion/runtime", request)
	if string(got.Data) != string(want.Data) || got.MediaType != want.MediaType {
		t.Fatalf("image = %+v", got)
	}
}
