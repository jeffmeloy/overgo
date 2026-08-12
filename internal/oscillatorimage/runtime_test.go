package oscillatorimage

import (
	"testing"

	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
)

type generatorFunc func(Request) (Image, error)

func (f generatorFunc) generate(request Request) (Image, error) { return f(request) }

func TestRegisteredRuntimeExecutesImageProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "image-model", recipe.TaskImageGen)
	want := Image{Pixels: []float32{0.25}, Channels: 1, Height: 1, Width: 1}
	if err := registerRuntime(fixture.Runtime, fixture.Model, generatorFunc(func(request Request) (Image, error) {
		if request.Class != 2 || request.Seed != 7 {
			t.Fatalf("request = %+v", request)
		}
		return want, nil
	})); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.ExecuteScalar("image/runtime", Request{Class: 2, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	datum, one := result.Outputs["image"].Single()
	got, typed := datum.Value.(Image)
	if !one || !typed || len(got.Pixels) != 1 || got.Pixels[0] != want.Pixels[0] || !result.Commit.Valid() {
		t.Fatalf("image = (%+v, %v, %v), commit=%v", got, one, typed, result.Commit)
	}
}

func TestValidateRequestRejectsNegativeClass(t *testing.T) {
	if err := ValidateRequest(Request{Class: -1}); err == nil {
		t.Fatal("negative class accepted")
	}
}
