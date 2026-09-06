package evaluation

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
)

// transcriptionRuntime keeps model execution separate from the shared target
// isolation, scoring, repetition and resource-recording loop.
type transcriptionRuntime interface {
	transcribe(context.Context, TranscriptionResourceInput, speechrecognition.RunBinding) (recipecontract.Transcription, runrecord.Run, error)
	close() error
}

type cpuTranscriptionRuntime struct {
	transcriber *speechrecognition.Transcriber
	workspace   speechrecognition.TranscriptionWorkspace
}

func loadTranscriptionRuntime(ctx context.Context, repository artifact.Repository, plan Plan, prompt string, maxTokens int, decodeRecipe artifact.ID, memoryBytes uint64) (transcriptionRuntime, error) {
	if prompt != "" {
		return loadProjectedTranscription(ctx, repository, plan, prompt, maxTokens, decodeRecipe)
	}
	transcriber, err := speechrecognition.LoadTranscriber(ctx, repository, plan.body.RuntimeRecipe, memoryBytes)
	if err != nil {
		return nil, err
	}
	return &cpuTranscriptionRuntime{transcriber: transcriber}, nil
}

func (runtime *cpuTranscriptionRuntime) transcribe(ctx context.Context, input TranscriptionResourceInput, binding speechrecognition.RunBinding) (recipecontract.Transcription, runrecord.Run, error) {
	return runtime.transcriber.Transcribe(ctx, input.Data, input.Origin, input.Policy, &runtime.workspace, binding)
}

func (*cpuTranscriptionRuntime) close() error { return nil }
