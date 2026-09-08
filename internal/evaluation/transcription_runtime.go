package evaluation

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
)

// transcriptionRuntime keeps model execution separate from the shared target
// isolation, scoring, repetition and resource-recording loop.
type transcriptionRuntime interface {
	transcribe(context.Context, TranscriptionResourceInput, []byte, speechrecognition.RunBinding) (recipecontract.Transcription, runrecord.Run, error)
	close(context.Context) error
}

type cpuTranscriptionRuntime struct {
	transcriber *speechrecognition.Session
}

func loadTranscriptionRuntime(ctx context.Context, repository artifact.Repository, plan Plan, prompt string, maxTokens int, decodeRecipe artifact.ID, memoryBytes uint64) (transcriptionRuntime, error) {
	if prompt != "" {
		return loadProjectedTranscription(ctx, repository, plan, prompt, maxTokens, decodeRecipe)
	}
	transcriber, err := speechrecognition.LoadSession(ctx, repository, plan.body.RuntimeRecipe, memoryBytes)
	if err != nil {
		return nil, err
	}
	return &cpuTranscriptionRuntime{transcriber: transcriber}, nil
}

func (runtime *cpuTranscriptionRuntime) transcribe(ctx context.Context, input TranscriptionResourceInput, data []byte, binding speechrecognition.RunBinding) (result recipecontract.Transcription, run runrecord.Run, err error) {
	lease, err := runtime.transcriber.Lease(ctx)
	if err != nil {
		return result, run, err
	}
	defer func() { err = errors.Join(err, lease.Release()) }()
	return lease.Transcribe(ctx, data, input.Reference.Origin, input.Policy, binding)
}

func (runtime *cpuTranscriptionRuntime) close(ctx context.Context) error {
	return runtime.transcriber.Close(ctx)
}
