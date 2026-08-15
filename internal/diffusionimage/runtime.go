package diffusionimage

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

type Request struct {
	Seed   int64 `json:"seed"`
	Steps  int   `json:"steps"`
	Height int   `json:"height"`
	Width  int   `json:"width"`
}

type samplePlan struct{ request Request }

type sampleFeatures struct {
	pixels        []float32
	height, width int
}

type generator interface {
	prepare(Request) (samplePlan, error)
	integrate(samplePlan) (sampleFeatures, error)
	decode(sampleFeatures) (latentimage.EncodedImage, error)
}

func ValidateRequest(request Request) error {
	if request.Steps <= 0 || request.Height <= 0 || request.Width <= 0 {
		return errors.New("diffusionimage: positive steps, height, and width required")
	}
	return nil
}

func (m *Model) prepare(request Request) (samplePlan, error) {
	if err := ValidateRequest(request); err != nil {
		return samplePlan{}, err
	}
	if m == nil {
		return samplePlan{}, errors.New("diffusionimage: model is unavailable")
	}
	tile := m.Cfg.PatchSize << uint(m.Cfg.NumLevels-1)
	if request.Height%tile != 0 || request.Width%tile != 0 {
		return samplePlan{}, errors.New("diffusionimage: image size is incompatible with artifact tile")
	}
	return samplePlan{request: request}, nil
}

func (m *Model) integrate(plan samplePlan) (sampleFeatures, error) {
	request := plan.request
	pixels, err := m.Sample(1, request.Height, request.Width, request.Steps, request.Seed)
	return sampleFeatures{pixels: pixels, height: request.Height, width: request.Width}, err
}

func (m *Model) decode(features sampleFeatures) (latentimage.EncodedImage, error) {
	return latentimage.EncodePlanarPNG(features.pixels, features.height, features.width)
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("diffusionimage: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, model)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model generator) error {
	return workflowruntime.RegisterPipeline(
		runtime, modelID,
		modelrecipe.ModuleDiffusionImagePrepare, model.prepare,
		modelrecipe.ModuleDiffusionImageIntegrate, model.integrate,
		modelrecipe.ModuleDiffusionImageDecode, model.decode,
		latentimage.PNGContent,
	)
}
