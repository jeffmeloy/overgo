// Package servingtest opens fixtures through the real recipe lifecycle.
package servingtest

import (
	"context"
	"errors"
	"os"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

// ResolveActiveGGUF: temporary RepoDB-backed serving fixture (capacity session).
func ResolveActiveGGUF(path string, placement recipe.Placement) (modelrecipe.LoadedProgram, error) {
	return ResolveActiveGGUFWithSession(path, placement, modelrecipe.DecodeSessionCapacity)
}

// ResolveActiveGGUFWithSession: serving fixture with an explicit decode-session
// policy, so tests can compare the capacity (append) path against the request
// (concat) path on the same model.
func ResolveActiveGGUFWithSession(
	path string, placement recipe.Placement, session modelrecipe.DecodeSessionPolicy,
) (modelrecipe.LoadedProgram, error) {
	root, err := os.MkdirTemp("", "overgo-serving-fixture-")
	if err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	defer os.RemoveAll(root)
	store, err := repodb.Open(root)
	if err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	defer store.Close()
	ctx := context.Background()
	file, err := gguf.Open(path)
	if err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	inventory, err := modelartifact.FromGGUF(file)
	if err != nil {
		_ = file.Close()
		return modelrecipe.LoadedProgram{}, err
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		_ = file.Close()
		return modelrecipe.LoadedProgram{}, err
	}
	profile, ok := model.LookupArchitecture(spec.Architecture)
	if !ok {
		_ = file.Close()
		return modelrecipe.LoadedProgram{}, errors.New("serving test: architecture profile is unavailable")
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		_ = file.Close()
		return modelrecipe.LoadedProgram{}, err
	}
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		return modelrecipe.LoadedProgram{}, errors.Join(err, closeErr)
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	prefix := "fixture/serving/" + inventory.Manifest.ID.String()
	if _, err := modelrecipe.PublishResolvedModelDefinition(
		ctx, store, prefix+"/facts", inventory, resolved,
	); err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		inventory.Manifest.ID, profileDocument.ID, document.ID, placement,
		session,
	)
	if err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, prefix+"/candidate", definition); err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, prefix+"/validated", definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, prefix+"/verification", definition.ID,
	)
	if err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, prefix+"/active", definition, verification, nil, nil,
	); err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	return modelrecipe.ResolveActiveGGUF(ctx, store, path)
}
