// Package servingtest opens fixtures through the real recipe lifecycle.
package servingtest

import (
	"context"
	"errors"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// ResolveActiveGGUFWithPolicy: serving fixture with exact session and residency identity.
func ResolveActiveGGUFWithPolicy(
	path string,
	placement recipe.Placement,
	session modelrecipe.DecodeSessionPolicy,
	residency recipe.ResidencyPolicy,
) (modelrecipe.LoadedProgram, error) {
	root, err := os.MkdirTemp("", "overgo-serving-fixture-")
	if err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	defer os.RemoveAll(root)
	store, err := overgodb.Open(root)
	if err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	defer store.Close()
	ctx := context.Background()
	if err := PublishActiveGGUFWithPolicy(ctx, store, path, placement, session, residency); err != nil {
		return modelrecipe.LoadedProgram{}, err
	}
	return modelrecipe.ResolveActiveGGUF(ctx, store, path)
}

// PublishActiveGGUFWithPolicy provisions one verified serving recipe.
func PublishActiveGGUFWithPolicy(
	ctx context.Context,
	store artifact.Repository,
	path string,
	placement recipe.Placement,
	session modelrecipe.DecodeSessionPolicy,
	residency recipe.ResidencyPolicy,
) error {
	file, err := gguf.Open(path)
	if err != nil {
		return err
	}
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		_ = file.Close()
		return err
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		_ = file.Close()
		return err
	}
	profile := spec.Profile()
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		_ = file.Close()
		return err
	}
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		return errors.Join(err, closeErr)
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		return err
	}
	prefix := "fixture/serving/" + inventory.Manifest.ID.String()
	if _, err := modelrecipe.PublishResolvedModelDefinition(
		ctx, store, inventory, resolved,
	); err != nil {
		return err
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		inventory.Manifest.ID, profileDocument.ID, document.ID, placement,
		session, residency,
	)
	if err != nil {
		return err
	}
	return modelrecipetest.PublishActivation(ctx, store, prefix, definition)
}
