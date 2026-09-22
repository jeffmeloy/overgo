package speechrecognition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

// transcriptContract names what a transcription execution produces, so the
// text leaves the runtime as a content-identified artifact rather than a
// bare value a run record cannot cite.
var transcriptContract = artifact.JSONContract(artifact.KindFile, "overgo.transcription-text.v1")

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
		}, transcriptContent)
}

// transcriptContent gives one transcript its content identity, the identity a
// run record cites and a later reader resolves the text by.
func transcriptContent(text string) (artifact.Content, error) {
	return artifact.JSONContent(transcriptContract, text)
}
