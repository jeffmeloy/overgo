package oscillatorimage

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var imageContract = artifact.JSONContract(artifact.KindOutput, "overgo.generated-image.v1")

type Request struct {
	Class int   `json:"class"`
	Seed  int64 `json:"seed"`
}

type planarImage struct {
	Pixels   []float32 `json:"pixels"`
	Channels int       `json:"channels"`
	Height   int       `json:"height"`
	Width    int       `json:"width"`
}

type generator interface {
	prepare(Request) (phasePlan, error)
	integrate(phasePlan) ([]float32, error)
	decode([]float32) (latentimage.EncodedImage, error)
}

func ValidateRequest(request Request) error {
	if request.Class < 0 {
		return errors.New("oscillatorimage: class must be non-negative")
	}
	return nil
}

func (m *Model) generate(request Request) (planarImage, error) {
	if err := ValidateRequest(request); err != nil {
		return planarImage{}, err
	}
	plan, err := m.prepare(request)
	if err != nil {
		return planarImage{}, err
	}
	features, err := m.integrate(plan)
	if err != nil {
		return planarImage{}, err
	}
	return m.decodePlanar(features)
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("oscillatorimage: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, model)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model generator) error {
	if err := workflowruntime.RegisterScalarStage(
		runtime, modelrecipe.ModuleOscillatorImagePrepare, modelID, model.prepare, nil,
	); err != nil {
		return err
	}
	if err := workflowruntime.RegisterScalarStage(
		runtime, modelrecipe.ModuleOscillatorImageIntegrate, modelID, model.integrate, nil,
	); err != nil {
		return err
	}
	return workflowruntime.RegisterJSONStage(runtime, modelrecipe.ModuleOscillatorImageDecode, modelID, imageContract, model.decode)
}
