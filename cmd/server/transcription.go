package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	llamaserver "overgo/internal/server"
	"overgo/internal/tokenizer"
)

// resolveTranscriptionServing selects declared native transcription without
// interpreting its weights as GGUF. A model with an inference activation keeps
// the existing inference launch path and its optional transcription workspace.
func resolveTranscriptionServing(ctx context.Context, repository, reference string) (*recipe.Definition, error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	entries, truncated, err := discovery.CapabilityCatalogForTasks(ctx, store, generationCatalogLimit, discovery.LoadMemo(ctx, store), recipe.TaskTranscription)
	if err != nil {
		return nil, err
	}
	entry, found, err := discovery.MatchReference(entries, truncated, reference)
	if err != nil || !found {
		return nil, err
	}
	if _, active, err := modelrecipe.ActiveRecord(ctx, store, entry.Model, recipe.TaskInference); err != nil || active {
		return nil, err
	}
	activation, _, err := modelrecipe.ResolveActiveCapability(ctx, store, entry.Model, recipe.TaskTranscription)
	if err != nil {
		return nil, err
	}
	return &activation.Definition, nil
}

type transcriptionRuntime struct {
	*llamaserver.TranscriptionWorkspace
	model artifact.ID
}

// ModelID binds HTTP observations and the served display alias to exact weights.
func (runtime *transcriptionRuntime) ModelID() artifact.ID { return runtime.model }

// Generate refuses the text-generation API for a transcription-only runtime.
func (*transcriptionRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("the selected model serves transcription, not text generation")
}

func serveTranscription(ctx context.Context, definition recipe.Definition, configured *llamaserver.TranscriptionPolicy, options serveOptions) error {
	var policy llamaserver.TranscriptionPolicy
	if configured == nil {
		var err error
		policy, err = llamaserver.LoadTranscriptionPolicy(filepath.Join(filepath.Dir(options.repository), transcriptionPolicyDocument))
		if err != nil {
			return fmt.Errorf("native transcription requires an explicit resource policy: %w", err)
		}
	} else {
		policy = *configured
	}
	if policy.Recipe.Valid() && policy.Recipe != definition.ID {
		return errors.New("transcription policy recipe does not match the selected model's active recipe")
	}
	policy.Recipe = definition.ID
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return fmt.Errorf("bind transcription source: %w", err)
	}
	store, err := overgodb.Open(options.repository)
	if err != nil {
		return err
	}
	defer store.Close()
	workspace, err := llamaserver.NewTranscriptionWorkspace(ctx, store, policy, commit)
	if err != nil {
		return err
	}
	defer workspace.Close(context.WithoutCancel(ctx))
	// The shared HTTP policy supplies transport bounds, not an inference
	// activation or an unrelated model. Numeric execution remains recipe-bound.
	hostPolicy, found, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		return errors.Join(errors.New("transcription server: HTTP policy unavailable"), err)
	}
	if options.modelID == "" {
		options.modelID = definition.Model.String()
	}
	handler, err := llamaserver.New(llamaserver.Config{
		RuntimePolicy: hostPolicy, ModelID: options.modelID, APIKey: options.apiKey,
		MaxConcurrent: options.maxConcurrent, RequestTimeout: options.requestTimeout,
		Repository: store, OvergoDBPath: options.repository, WebUIDir: options.webuiDir,
		LibraryIntake: serverLibraryIntake(), ProviderKeys: providerIntake.Keys,
	}, &transcriptionRuntime{TranscriptionWorkspace: workspace, model: definition.Model})
	if err != nil {
		return err
	}
	defer handler.Close()
	log.Printf("serving transcription model %s recipe %s on http://%s", definition.Model, definition.ID, options.address)
	return serve(ctx, options.address, handler)
}
