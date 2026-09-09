// Package providerintake binds the hosted-provider intake the server's
// library routes call: a declaration, a listing, a retirement and a key
// commit exactly as the provider command performs them, under the
// executable's source commit. The launcher and the swap proxy's idle
// shell share it, so a declaration from the page is the same store claim
// wherever the page is served from.
package providerintake

import (
	"context"
	"strings"

	"overgo/internal/overgodb"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay"
	"overgo/internal/runrecord"
	"overgo/internal/server"
)

// Intake is the provider intake over one catalog bound.
type Intake struct {
	// CatalogLimit bounds the store's servable listing a reference resolves against.
	CatalogLimit int
}

// Declare commits the page's declaration exactly as the provider command's
// file does, under this executable's source commit, and answers each
// declared model with the refusal its key's absence carries now.
func (intake Intake) Declare(ctx context.Context, store *overgodb.Store, declaration server.ProviderDeclaration) ([]server.DeclaredProvider, error) {
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return nil, err
	}
	declarations, err := remoteprovider.DeclareDocument(ctx, store, remoteprovider.Document{
		Name: declaration.Name, Endpoint: declaration.Endpoint, KeyEnvironment: declaration.KeyEnvironment,
		Models: declaration.Models, ContextLength: declaration.ContextLength,
	}, commit)
	declared := make([]server.DeclaredProvider, 0, len(declarations))
	for _, declaration := range declarations {
		declared = append(declared, server.DeclaredProvider{
			Location: declaration.Location, Model: declaration.Model.String(), Recipe: declaration.Recipe.ID.String(),
			Refusal: remoteprovider.Refusal(declaration.Provider),
		})
	}
	return declared, err
}

// ListModels asks the provider at the endpoint for its models under the
// key its variable holds.
func (intake Intake) ListModels(ctx context.Context, endpoint, keyEnvironment string) ([]server.ProviderModel, error) {
	listed, err := remoterelay.ListModels(ctx, remoteprovider.Provider{Endpoint: strings.TrimRight(endpoint, "/"), KeyEnvironment: keyEnvironment}, nil)
	if err != nil {
		return nil, err
	}
	models := make([]server.ProviderModel, 0, len(listed))
	for _, model := range listed {
		models = append(models, server.ProviderModel{ID: model.ID, Name: model.Name, ContextLength: model.ContextLength})
	}
	return models, nil
}

// Retire retires the declared hosted model at the location exactly as the
// provider command does, under this executable's source commit.
func (intake Intake) Retire(ctx context.Context, store *overgodb.Store, location, reason string) error {
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return err
	}
	return remoteprovider.Retire(ctx, store, intake.CatalogLimit, location, commit, reason)
}

// Keys places the key the page enters for a hosted model in this
// process's environment (never on disk); the variable's name comes back.
func (intake Intake) Keys(ctx context.Context, store *overgodb.Store, reference, key string) (string, error) {
	provider, err := remoteprovider.SetKey(ctx, store, intake.CatalogLimit, reference, key)
	return provider.KeyEnvironment, err
}

// Library is the server's library intake over this provider intake, with
// the local model intake's file and registration steps; the validation
// step, which needs a served model, is the caller's.
func (intake Intake) Library(modelFiles func(path, explicitProjector string) (string, string, error), register func(context.Context, *overgodb.Store, string, string) (map[string]any, error)) server.LibraryIntake {
	return server.LibraryIntake{ModelFiles: modelFiles, Register: register, DeclareProvider: intake.Declare, ListProviderModels: intake.ListModels, RetireProvider: intake.Retire}
}
