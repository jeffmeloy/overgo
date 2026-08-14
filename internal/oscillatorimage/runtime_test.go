package oscillatorimage

import (
	"testing"

	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
)

type generatorFunc struct {
	prepareFunc   func(Request) (phasePlan, error)
	integrateFunc func(phasePlan) ([]float32, error)
	decodeFunc    func([]float32) (Image, error)
}

func (f generatorFunc) prepare(request Request) (phasePlan, error)  { return f.prepareFunc(request) }
func (f generatorFunc) integrate(plan phasePlan) ([]float32, error) { return f.integrateFunc(plan) }
func (f generatorFunc) decode(features []float32) (Image, error)    { return f.decodeFunc(features) }

func TestRegisteredRuntimeExecutesImageProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "image-model", recipe.TaskImageGen)
	want := Image{Pixels: []float32{0.25}, Channels: 1, Height: 1, Width: 1}
	stages := generatorFunc{
		prepareFunc: func(request Request) (phasePlan, error) {
			if request.Class != 2 || request.Seed != 7 {
				t.Fatalf("request = %+v", request)
			}
			return phasePlan{state: []float32{1}}, nil
		},
		integrateFunc: func(plan phasePlan) ([]float32, error) {
			if len(plan.state) != 1 || plan.state[0] != 1 {
				t.Fatalf("phase plan = %+v", plan)
			}
			return []float32{0.25}, nil
		},
		decodeFunc: func(features []float32) (Image, error) {
			if len(features) != 1 || features[0] != want.Pixels[0] {
				t.Fatalf("features = %v", features)
			}
			return want, nil
		},
	}
	if err := registerRuntime(fixture.Runtime, fixture.Model, stages); err != nil {
		t.Fatal(err)
	}
	got := modelrecipetest.MustExecuteScalar[Image](
		t, fixture, "image/runtime", Request{Class: 2, Seed: 7},
	)
	if len(got.Pixels) != 1 || got.Pixels[0] != want.Pixels[0] {
		t.Fatalf("image = %+v", got)
	}
}

func TestValidateRequestRejectsNegativeClass(t *testing.T) {
	if err := ValidateRequest(Request{Class: -1}); err == nil {
		t.Fatal("negative class accepted")
	}
}
