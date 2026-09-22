package speechrecognition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

// RegisterRuntime binds a loaded recognizer to the transcription stage of one
// model's program, so a capability execution reaches this runtime by the
// module the audio contract declares and by nothing else.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, recognizer *Recognizer) error {
	if recognizer == nil {
		return errors.New("speechrecognition: incomplete runtime binding")
	}
	return workflowruntime.RegisterContextStage(runtime, modelrecipe.ModuleTranscribeAudio, modelID,
		func(ctx context.Context, request TranscriptionRequest) (string, error) {
			if err := context.Cause(ctx); err != nil {
				return "", err
			}
			return recognizer.Transcribe(ctx, request)
		}, nil)
}
