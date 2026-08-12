package oscillatorimage

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var imageContract = artifact.JSONContract(artifact.KindOutput, "overgo.generated-image.v1")

type Request struct {
	Class int   `json:"class"`
	Seed  int64 `json:"seed"`
}

type Image struct {
	Pixels   []float32 `json:"pixels"`
	Channels int       `json:"channels"`
	Height   int       `json:"height"`
	Width    int       `json:"width"`
}

type generator interface {
	generate(Request) (Image, error)
}

func ValidateRequest(request Request) error {
	if request.Class < 0 {
		return errors.New("oscillatorimage: class must be non-negative")
	}
	return nil
}

func (m *Model) generate(request Request) (Image, error) {
	if m == nil {
		return Image{}, errors.New("oscillatorimage: model is unavailable")
	}
	if err := ValidateRequest(request); err != nil {
		return Image{}, err
	}
	pixels, err := m.Generate(request.Class, request.Seed)
	if err != nil {
		return Image{}, err
	}
	return Image{
		Pixels: pixels, Channels: m.Cfg.OutChannels,
		Height: m.Cfg.OutH(), Width: m.Cfg.OutW(),
	}, nil
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("oscillatorimage: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, model)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model generator) error {
	return workflowruntime.RegisterJSONStage(
		runtime, modelrecipe.ModuleImageGenerate, modelID, imageContract, model.generate,
	)
}
