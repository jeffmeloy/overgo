package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"overgo/internal/libraryintake"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay"
	llamaserver "overgo/internal/server"
)

// remoteServing is a served remote model: the declaration the store
// holds, its active recipe and the recipe's policy.
type remoteServing struct {
	provider   remoteprovider.Provider
	definition recipe.Definition
	policy     modelrecipe.RuntimePolicy
}

// resolveRemoteServing reads the served reference against the store's
// remote declarations: a remote location or a remote model's identity
// serves through the relay, and any other reference is a local model
// path, reported as nil.
func resolveRemoteServing(ctx context.Context, repository, reference string) (*remoteServing, error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, fmt.Errorf("open model recipe repository: %w", err)
	}
	defer store.Close()
	provider, modelID, remote, err := remoteprovider.Reference(ctx, store, generationCatalogLimit, reference)
	if err != nil || !remote {
		return nil, err
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil {
		return nil, err
	}
	if !active {
		return nil, fmt.Errorf("remote model %s has no active inference recipe", reference)
	}
	policy, err := modelrecipe.ResolveRuntimePolicy(ctx, store, activation.Definition)
	if err != nil {
		return nil, err
	}
	return &remoteServing{provider: provider, definition: activation.Definition, policy: policy}, nil
}

// remoteServeOptions are the launch flags a remote serving honours.
type remoteServeOptions struct {
	address, apiKey, modelID                                      string
	maxTokens, maxConcurrent, storedResponses, responseStoreBytes int
	requestTimeout                                                time.Duration
	repository, hubRoot, webuiDir                                 string
}

// remoteRuntime is the relay with the store's workflow workspaces beside
// it, the way the local runtime carries the runner.
type remoteRuntime struct {
	*remoterelay.Generator
	llamaserver.WorkflowWorkspaceAPI
}

// serveRemote serves the declared model through the relay: the store's
// generation and library workspaces ride along as they do for a local
// model, the environment records the remote backend so every interaction
// and observation under it carries the non-reproducible mark, and the
// key's absence refuses the launch by the variable's name.
func serveRemote(ctx context.Context, remote *remoteServing, options remoteServeOptions) error {
	generator, err := remoterelay.New(remote.provider, remote.definition, nil)
	if err != nil {
		return err
	}
	environment, err := remoteprovider.Environment(remote.provider)
	if err != nil {
		return err
	}
	workspaceStore, err := overgodb.Open(options.repository)
	if err != nil {
		return fmt.Errorf("open workspace repository: %w", err)
	}
	defer workspaceStore.Close()
	generation := llamaserver.NewStoreGenerationWorkspace(workspaceStore,
		llamaserver.BindGenerationCatalog(mediacapability.Catalog, mediacapability.Controls, mediacapability.OutputContent), generationCatalogLimit)
	handler, err := llamaserver.New(llamaserver.Config{
		RuntimePolicy: remote.policy, ModelID: options.modelID, MaxTokens: options.maxTokens, MaxConcurrent: options.maxConcurrent,
		APIKey: options.apiKey, RequestTimeout: options.requestTimeout,
		MaxStoredResponses: options.storedResponses, ResponseStoreBytes: options.responseStoreBytes,
		OvergoDBPath: options.repository, Repository: workspaceStore, Environment: environment,
		HubToken: os.Getenv("OVERGO_HF_TOKEN"), HubDownloadRoot: options.hubRoot, WebUIDir: options.webuiDir,
		LibraryIntake: llamaserver.LibraryIntake{ModelFiles: libraryintake.ModelFiles, Register: libraryintake.Register, Validate: libraryintake.Validate},
	}, &remoteRuntime{Generator: generator, WorkflowWorkspaceAPI: llamaserver.WorkflowWorkspaceSet{generation}})
	if err != nil {
		return err
	}
	defer handler.Close()
	log.Printf("serving remote model %q at %s on http://%s", options.modelID, remote.provider.Name, options.address)
	return serve(ctx, options.address, handler)
}
