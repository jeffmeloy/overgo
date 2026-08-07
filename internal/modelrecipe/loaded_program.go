package modelrecipe

import (
	"context"
	"errors"
	"reflect"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
)

// LoadedProgram: sealed verified GGUF ownership.
type LoadedProgram struct {
	state *loadedProgramState
}

type loadedProgramState struct {
	File         *gguf.File
	Path         string
	Spec         model.Spec
	Weights      model.Weights
	Inventory    modelartifact.Inventory
	EvidenceTier recipe.EvidenceTier
	Program      Plan
}

// ConsumedProgram: one-shot serving ownership.
type ConsumedProgram struct {
	state *loadedProgramState
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
	activation, active, err := ActiveRecord(ctx, store, loaded.state.Inventory.Manifest.ID, recipe.TaskInference)
	if err != nil {
		return fail(err)
	}
	if !active {
		return fail(errors.New("model recipe: active inference recipe is absent"))
	}
	definition := activation.Definition
	loaded.state.EvidenceTier = activation.Tier
	definitionID, ok := definition.Dependency(recipe.DependencyDefinition, 0)
	if !ok {
		return fail(errors.New("model recipe: active recipe has no model definition"))
	}
	resolved, err := ResolveModelDefinition(ctx, store, definitionID)
	if err != nil {
		return fail(err)
	}
	if resolved.Document.Model != loaded.state.Inventory.Manifest.ID ||
		resolved.Tensors.ID != loaded.state.Inventory.TensorInventory.ID {
		return fail(errors.New("model recipe: loaded GGUF differs from active model definition"))
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
	loaded := LoadedProgram{state: &loadedProgramState{File: file, Path: path}}
	fail := func(cause error) (LoadedProgram, error) {
		return LoadedProgram{}, errors.Join(cause, file.Close())
	}
	loaded.state.Inventory, err = modelartifact.FromGGUF(file)
	if err != nil {
		return fail(err)
	}
	loaded.state.Spec, err = model.ReadSpec(file)
	if err != nil {
		return fail(err)
	}
	return loaded, nil
}

func (l *LoadedProgram) bindResolved(
	definition recipe.Definition,
	resolved ResolvedModelDefinition,
) error {
	if l == nil || l.state == nil || l.state.File == nil {
		return errors.New("model recipe: loaded GGUF is unavailable")
	}
	observed, err := model.ReadSpecWithProfile(l.state.File, resolved.Profile.Policy)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(observed, resolved.Spec) {
		return errors.New("model recipe: loaded GGUF specification differs from model definition")
	}
	if err := validateProgramIdentity(definition, programIdentity(definition)); err != nil {
		return err
	}
	weights, err := model.ReadWeights(l.state.File, observed)
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
	if observed.Architecture != program.Model.Profile.Name {
		return errors.New("model recipe: loaded program architecture differs")
	}
	l.state.Spec, l.state.Weights, l.state.Program = observed, weights, program
	l.state.Inventory = modelartifact.Inventory{}
	return nil
}

// Identity: immutable compiled serving identity.
func (l LoadedProgram) Identity() (ProgramIdentity, bool) {
	if l.state == nil {
		return ProgramIdentity{}, false
	}
	return l.state.Program.Identity, true
}

// Architecture: verified model architecture.
func (l LoadedProgram) Architecture() (string, bool) {
	if l.state == nil {
		return "", false
	}
	return l.state.Spec.Architecture, true
}

// LayerPlans: immutable compiled layer-program copy.
func (l LoadedProgram) LayerPlans() ([]model.LayerPlan, bool) {
	if l.state == nil {
		return nil, false
	}
	return append([]model.LayerPlan(nil), l.state.Program.Model.Layers...), true
}

// Consume: transfers resolved serving ownership exactly once.
func (l *LoadedProgram) Consume() (*ConsumedProgram, error) {
	if l == nil || l.state == nil || l.state.File == nil {
		return nil, errors.New("model recipe: loaded program is unavailable or consumed")
	}
	state := l.state
	l.state = nil
	return &ConsumedProgram{state: state}, nil
}

func (p *ConsumedProgram) File() *gguf.File                  { return p.state.File }
func (p *ConsumedProgram) Path() string                      { return p.state.Path }
func (p *ConsumedProgram) Spec() model.Spec                  { return p.state.Spec }
func (p *ConsumedProgram) Weights() model.Weights            { return p.state.Weights }
func (p *ConsumedProgram) Plan() Plan                        { return p.state.Program }
func (p *ConsumedProgram) EvidenceTier() recipe.EvidenceTier { return p.state.EvidenceTier }

// Disown: transfers file lifetime to the serving runtime.
func (p *ConsumedProgram) Disown() {
	if p != nil {
		p.state = nil
	}
}

// Close: releases consumed ownership before runtime transfer.
func (p *ConsumedProgram) Close() error {
	if p == nil || p.state == nil || p.state.File == nil {
		return nil
	}
	err := p.state.File.Close()
	p.state = nil
	return err
}

// Close: releases untransferred GGUF ownership.
func (l *LoadedProgram) Close() error {
	if l == nil || l.state == nil || l.state.File == nil {
		return nil
	}
	err := l.state.File.Close()
	l.state = nil
	return err
}
