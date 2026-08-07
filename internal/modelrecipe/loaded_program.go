package modelrecipe

import (
	"context"
	"errors"
	"fmt"
	"reflect"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
)

// LoadedProgram: verified GGUF plus authoritative recipe program.
type LoadedProgram struct {
	File       *gguf.File
	Path       string
	Spec       model.Spec
	Weights    model.Weights
	Inventory  modelartifact.Inventory
	Profile    ProfileDocument
	Definition ModelDefinitionDocument
	Program    Plan
}

// ResolveActiveGGUF: verifies and compiles the active recipe before execution.
func ResolveActiveGGUF(
	ctx context.Context,
	store artifact.Reader,
	path string,
) (LoadedProgram, error) {
	loaded, err := loadGGUFFacts(path)
	if err != nil {
		return LoadedProgram{}, err
	}
	fail := func(cause error) (LoadedProgram, error) {
		return LoadedProgram{}, errors.Join(cause, loaded.Close())
	}
	definition, active, err := Active(ctx, store, loaded.Inventory.Manifest.ID, recipe.TaskInference)
	if err != nil {
		return fail(err)
	}
	if !active {
		return fail(errors.New("model recipe: active inference recipe is absent"))
	}
	definitionID, ok := definition.Dependency(recipe.DependencyDefinition, 0)
	if !ok {
		return fail(errors.New("model recipe: active recipe has no model definition"))
	}
	resolved, err := ResolveModelDefinition(ctx, store, definitionID)
	if err != nil {
		return fail(err)
	}
	if resolved.Document.Model != loaded.Inventory.Manifest.ID ||
		resolved.Tensors.ID != loaded.Inventory.TensorInventory.ID {
		return fail(errors.New("model recipe: loaded GGUF differs from active model definition"))
	}
	if err := loaded.bindResolved(definition, resolved); err != nil {
		return fail(err)
	}
	return loaded, nil
}

// LoadFixtureGGUF: identity-complete program without lifecycle activation.
func LoadFixtureGGUF(path string, placement recipe.Placement) (LoadedProgram, error) {
	loaded, err := loadGGUFFacts(path)
	if err != nil {
		return LoadedProgram{}, err
	}
	fail := func(cause error) (LoadedProgram, error) {
		return LoadedProgram{}, errors.Join(cause, loaded.Close())
	}
	profile, err := NewProfileDocument(loaded.Spec.Profile())
	if err != nil {
		return fail(err)
	}
	document, err := NewModelDefinitionDocument(profile, loaded.Inventory.TensorInventory, loaded.Spec)
	if err != nil {
		return fail(err)
	}
	resolved, err := document.Resolve(profile, loaded.Inventory.TensorInventory)
	if err != nil {
		return fail(err)
	}
	definition, err := InferenceWithModelDefinition(
		loaded.Inventory.Manifest.ID, profile.ID, document.ID, placement,
	)
	if err != nil {
		return fail(err)
	}
	if err := loaded.bindResolved(definition, resolved); err != nil {
		return fail(err)
	}
	return loaded, nil
}

func loadGGUFFacts(path string) (LoadedProgram, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return LoadedProgram{}, err
	}
	loaded := LoadedProgram{File: file, Path: path}
	fail := func(cause error) (LoadedProgram, error) {
		return LoadedProgram{}, errors.Join(cause, file.Close())
	}
	loaded.Inventory, err = modelartifact.FromGGUF(file)
	if err != nil {
		return fail(err)
	}
	loaded.Spec, err = model.ReadSpec(file)
	if err != nil {
		return fail(err)
	}
	return loaded, nil
}

func (l *LoadedProgram) bindResolved(
	definition recipe.Definition,
	resolved ResolvedModelDefinition,
) error {
	if l == nil || l.File == nil {
		return errors.New("model recipe: loaded GGUF is unavailable")
	}
	observed, err := model.ReadSpecWithProfile(l.File, resolved.Profile.Policy)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(observed, resolved.Spec) {
		return errors.New("model recipe: loaded GGUF specification differs from model definition")
	}
	if err := validateProgramIdentity(definition, programIdentity(definition)); err != nil {
		return err
	}
	weights, err := model.ReadWeights(l.File, observed)
	if err != nil {
		return err
	}
	program, err := CompileModelDefinition(definition, resolved, weights)
	if err != nil {
		return err
	}
	if err := program.ValidateServing(); err != nil {
		return err
	}
	l.Spec, l.Weights = observed, weights
	l.Profile, l.Definition, l.Program = resolved.Profile, resolved.Document, program
	return l.Validate()
}

// Validate: loaded artifacts match compiled identity.
func (l LoadedProgram) Validate() error {
	if l.File == nil || l.Path == "" {
		return errors.New("model recipe: loaded program source is unavailable")
	}
	if err := l.Program.ValidateServing(); err != nil {
		return err
	}
	identity := l.Program.Identity
	profileID, profileOK := l.Program.Recipe.Dependency(recipe.DependencyProfile, 0)
	definitionID, definitionOK := l.Program.Recipe.Dependency(recipe.DependencyDefinition, 0)
	if identity.Model != l.Inventory.Manifest.ID || !profileOK || !definitionOK ||
		identity.Profile != profileID || identity.Definition != definitionID ||
		l.Profile.ID != profileID || l.Definition.ID != definitionID {
		return errors.New("model recipe: loaded program identity mismatch")
	}
	resolved, err := l.Definition.Resolve(l.Profile, l.Inventory.TensorInventory)
	if err != nil {
		return err
	}
	observedSpec, err := model.ReadSpecWithProfile(l.File, l.Profile.Policy)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(observedSpec, resolved.Spec) || !reflect.DeepEqual(observedSpec, l.Spec) {
		return errors.New("model recipe: loaded specification changed after resolution")
	}
	observedWeights, err := model.ReadWeights(l.File, observedSpec)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(observedWeights, l.Weights) {
		return errors.New("model recipe: loaded weight catalog changed after resolution")
	}
	expected, err := model.CompileModelPlanWithProfile(observedSpec, observedWeights, l.Profile.Policy)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(expected, l.Program.Model) {
		return errors.New("model recipe: compiled model program differs from resolved facts")
	}
	if l.Spec.Architecture != l.Program.Model.Profile.Name {
		return fmt.Errorf("model recipe: loaded program shape differs for %q", l.Spec.Architecture)
	}
	return nil
}

// Close: releases untransferred GGUF ownership.
func (l *LoadedProgram) Close() error {
	if l == nil || l.File == nil {
		return nil
	}
	err := l.File.Close()
	l.File = nil
	return err
}
