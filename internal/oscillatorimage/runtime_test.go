package oscillatorimage

import (
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
)

type generatorFunc struct {
	prepareFunc   func(Request) (phasePlan, error)
	integrateFunc func(phasePlan) ([]float32, error)
	decodeFunc    func([]float32) (latentimage.EncodedImage, error)
}

type videoGeneratorFunc struct {
	prepareFunc   func(VideoRequest) (videoPlan, error)
	integrateFunc func(videoPlan) (videoFeatures, error)
	decodeFunc    func(videoFeatures) (EncodedVideo, error)
}

func (f videoGeneratorFunc) prepareVideo(request VideoRequest) (videoPlan, error) {
	return f.prepareFunc(request)
}
func (f videoGeneratorFunc) integrateVideo(plan videoPlan) (videoFeatures, error) {
	return f.integrateFunc(plan)
}
func (f videoGeneratorFunc) decodeVideo(features videoFeatures) (EncodedVideo, error) {
	return f.decodeFunc(features)
}

func (f generatorFunc) prepare(request Request) (phasePlan, error)  { return f.prepareFunc(request) }
func (f generatorFunc) integrate(plan phasePlan) ([]float32, error) { return f.integrateFunc(plan) }
func (f generatorFunc) decode(features []float32) (latentimage.EncodedImage, error) {
	return f.decodeFunc(features)
}

func TestRegisteredRuntimeExecutesImageProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "image-model", recipe.TaskImageGen)
	want := latentimage.EncodedImage{Data: []byte{1, 2, 3}, MediaType: "image/png", Channels: 3, Height: 1, Width: 1}
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
		decodeFunc: func(features []float32) (latentimage.EncodedImage, error) {
			if len(features) != 1 || features[0] != 0.25 {
				t.Fatalf("features = %v", features)
			}
			return want, nil
		},
	}
	if err := registerRuntime(fixture.Runtime, fixture.Model, stages); err != nil {
		t.Fatal(err)
	}
	got := modelrecipetest.MustExecuteScalar[latentimage.EncodedImage](
		t, fixture, "image/runtime", Request{Class: 2, Seed: 7},
	)
	if string(got.Data) != string(want.Data) || got.MediaType != want.MediaType {
		t.Fatalf("image = %+v", got)
	}
}

func TestValidateRequestRejectsNegativeClass(t *testing.T) {
	if err := ValidateRequest(Request{Class: -1}); err == nil {
		t.Fatal("negative class accepted")
	}
}

func TestRegisteredRuntimeExecutesVideoProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "video-model", recipe.TaskVideoGen)
	want := EncodedVideo{Data: []byte{4, 5, 6}, MediaType: "image/gif", Frames: 2, Channels: 3, Height: 8, Width: 8, ChangedPixels: 1}
	stages := videoGeneratorFunc{
		prepareFunc: func(request VideoRequest) (videoPlan, error) {
			if request != (VideoRequest{Class: 2, Seed: 7, Frames: 2, Scale: 4}) {
				t.Fatalf("request = %+v", request)
			}
			return videoPlan{request: request}, nil
		},
		integrateFunc: func(plan videoPlan) (videoFeatures, error) {
			return videoFeatures{frames: [][]float32{{0.25}, {0.5}}, scale: plan.request.Scale}, nil
		},
		decodeFunc: func(features videoFeatures) (EncodedVideo, error) {
			if len(features.frames) != 2 || features.scale != 4 {
				t.Fatalf("features = %+v", features)
			}
			return want, nil
		},
	}
	if err := registerVideoRuntime(fixture.Runtime, fixture.Model, stages); err != nil {
		t.Fatal(err)
	}
	got := modelrecipetest.MustExecuteScalar[EncodedVideo](
		t, fixture, "video/runtime", VideoRequest{Class: 2, Seed: 7, Frames: 2, Scale: 4},
	)
	if string(got.Data) != string(want.Data) || got.MediaType != want.MediaType || got.Frames != want.Frames {
		t.Fatalf("video = %+v", got)
	}
}
