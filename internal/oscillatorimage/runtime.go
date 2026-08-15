package oscillatorimage

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var videoContract = artifact.JSONContract(artifact.KindOutput, "overgo.generated-video.v1")

type videoGenerator interface {
	prepareVideo(VideoRequest) (videoPlan, error)
	integrateVideo(videoPlan) (videoFeatures, error)
	decodeVideo(videoFeatures) (EncodedVideo, error)
}

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

func RegisterVideoRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("oscillatorimage: incomplete video runtime binding")
	}
	return registerVideoRuntime(runtime, modelID, model)
}

func registerVideoRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model videoGenerator) error {
	return workflowruntime.RegisterJSONPipeline(
		runtime, modelID, videoContract,
		modelrecipe.ModuleOscillatorVideoPrepare, model.prepareVideo,
		modelrecipe.ModuleOscillatorVideoIntegrate, model.integrateVideo,
		modelrecipe.ModuleOscillatorVideoDecode, model.decodeVideo,
	)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model generator) error {
	return workflowruntime.RegisterPipeline(
		runtime, modelID,
		modelrecipe.ModuleOscillatorImagePrepare, model.prepare,
		modelrecipe.ModuleOscillatorImageIntegrate, model.integrate,
		modelrecipe.ModuleOscillatorImageDecode, model.decode,
		latentimage.PNGContent,
	)
}
