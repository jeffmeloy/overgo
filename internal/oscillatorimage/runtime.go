package oscillatorimage

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var videoContract = artifact.JSONContract(artifact.KindOutput, "overgo.generated-video.v1")

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

func ValidateRequest(request Request) error {
	if request.Class < 0 {
		return errors.New("oscillatorimage: class must be non-negative")
	}
	return nil
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("oscillatorimage: incomplete runtime binding")
	}
	if err := workflowruntime.RegisterContextPipeline(
		runtime, modelID,
		modelrecipe.ModuleOscillatorImagePrepare, model.prepare,
		modelrecipe.ModuleOscillatorImageIntegrate, model.integrate,
		modelrecipe.ModuleOscillatorImageDecode, model.decode,
		latentimage.PNGContent,
	); err != nil {
		return err
	}
	return workflowruntime.RegisterContextPipeline(
		runtime, modelID,
		modelrecipe.ModuleOscillatorVideoPrepare, model.prepareVideo,
		modelrecipe.ModuleOscillatorVideoIntegrate, model.integrateVideo,
		modelrecipe.ModuleOscillatorVideoDecode, model.decodeVideo,
		func(video EncodedVideo) (artifact.Content, error) { return artifact.JSONContent(videoContract, video) },
	)
}
