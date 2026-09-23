package vitencoder

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var embeddingContract = artifact.JSONContract(artifact.KindOutput, "overgo.image-embedding-output.v1")

// EmbeddingRequest is one encoded image, PNG or JPEG.
type EmbeddingRequest struct {
	Image []byte `json:"image"`
}

// ValidateEmbeddingRequest refuses a request without an image before any
// model is loaded.
func ValidateEmbeddingRequest(request EmbeddingRequest) error {
	if len(request.Image) == 0 {
		return errors.New("vitencoder: embedding requires an encoded image")
	}
	return nil
}

// RegisterRuntime binds one loaded encoder to its embedding stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, encoder *Encoder) error {
	if encoder == nil {
		return errors.New("vitencoder: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONStage[EmbeddingRequest, []float32](
		runtime, modelrecipe.ModuleEmbedImage, modelID, embeddingContract,
		func(request EmbeddingRequest) ([]float32, error) {
			if err := ValidateEmbeddingRequest(request); err != nil {
				return nil, err
			}
			return encoder.Embed(request.Image)
		},
	)
}
