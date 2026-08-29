// Package modelmerge owns offline weight operations that require exact model
// definition and tensor-inventory compatibility.
package modelmerge

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/tensor"
)

// Weight is one materialized F32 tensor with exact layout.
type Weight struct {
	Layout tensor.Shape `json:"layout"`
	Values []float32    `json:"values"`
}

// Snapshot is an immutable offline model tensor set. Base is present only for
// a fine-tuned or task-arithmetic derivative.
type Snapshot struct {
	ID         artifact.ID       `json:"-"`
	Definition artifact.ID       `json:"definition"`
	Base       artifact.ID       `json:"base,omitzero"`
	Tensors    map[string]Weight `json:"tensors"`
}

// Selection chooses one complete compatible tensor from one source model.
type Selection struct {
	Tensor string
	Source artifact.ID
}

// TaskDelta is a same-base fine-tuned model with a finite arithmetic scale.
type TaskDelta struct {
	Model Snapshot
	Scale float32
}

// Result carries the sealed model and its exact source lineage.
type Result struct {
	Model   Snapshot
	Lineage []artifact.Lineage
}

// Compiler has no runtime graph authority; its zero value seals and combines
// offline F32 snapshots.
type Compiler struct{}

var weightElements = tensor.Shape.Elements

// Seal creates a content-addressed snapshot after validating every tensor.
func (compiler Compiler) Seal(
	definition, base artifact.ID,
	weights map[string]Weight,
) (Snapshot, error) {
	snapshot := Snapshot{Definition: definition, Base: base, Tensors: cloneWeights(weights)}
	id, err := compiler.identity(snapshot)
	if err != nil {
		return Snapshot{}, err
	}
	snapshot.ID = id
	return snapshot, nil
}

// ValidateIdentity verifies the complete tensor content and model identity.
func (snapshot Snapshot) ValidateIdentity() error {
	want, err := (Compiler{}).identity(snapshot)
	if err != nil {
		return err
	}
	if snapshot.ID != want {
		return errors.New("model merge: snapshot identity differs")
	}
	return nil
}

// Passthrough selects whole tensors only after every source proves the same
// model definition and exact inventory geometry.
func (compiler Compiler) Passthrough(sources []Snapshot, selections []Selection) (Result, error) {
	if !checked.Nonzero(len(sources)) {
		return Result{}, errors.New("model merge: passthrough sources are empty")
	}
	for _, source := range sources {
		if err := source.ValidateIdentity(); err != nil {
			return Result{}, err
		}
	}
	reference := sources[tensor.FirstOffset]
	for _, source := range sources[tensor.SingletonExtent:] {
		if err := compatibleInventory(reference, source); err != nil {
			return Result{}, err
		}
	}
	byID := make(map[artifact.ID]Snapshot, len(sources))
	for _, source := range sources {
		byID[source.ID] = source
	}
	selected := make(map[string]Weight, len(selections))
	for _, selection := range selections {
		source, ok := byID[selection.Source]
		weight, found := source.Tensors[selection.Tensor]
		if !ok || !found {
			return Result{}, errors.New("model merge: passthrough selection is absent")
		}
		if _, duplicate := selected[selection.Tensor]; duplicate {
			return Result{}, errors.New("model merge: passthrough tensor is selected more than once")
		}
		selected[selection.Tensor] = cloneWeight(weight)
	}
	if len(selected) != len(reference.Tensors) {
		return Result{}, errors.New("model merge: passthrough does not cover the exact tensor inventory")
	}
	model, err := compiler.Seal(reference.Definition, artifact.ID{}, selected)
	if err != nil {
		return Result{}, err
	}
	parents := make([]artifact.ID, tensor.FirstOffset, len(sources))
	for _, source := range sources {
		parents = append(parents, source.ID)
	}
	return Result{Model: model, Lineage: artifact.DependencyLineage(model.ID, parents...)}, nil
}

// TaskArithmetic computes base + sum(scale * (fine-tuned - base)) and admits
// only fine-tuned snapshots that bind the same exact base identity.
func (compiler Compiler) TaskArithmetic(base Snapshot, tasks []TaskDelta) (Result, error) {
	if err := base.ValidateIdentity(); err != nil {
		return Result{}, err
	}
	if base.Base.Valid() || !checked.Nonzero(len(tasks)) {
		return Result{}, errors.New("model merge: task arithmetic base or task set is invalid")
	}
	output := cloneWeights(base.Tensors)
	parents := []artifact.ID{base.ID}
	for _, task := range tasks {
		if err := task.Model.ValidateIdentity(); err != nil {
			return Result{}, err
		}
		if task.Model.Base != base.ID || !checked.Finite32(task.Scale) {
			return Result{}, errors.New("model merge: task does not bind the exact base or finite scale")
		}
		if err := compatibleInventory(base, task.Model); err != nil {
			return Result{}, err
		}
		for name, baseWeight := range base.Tensors {
			combined := output[name]
			fineTuned := task.Model.Tensors[name]
			for index := range combined.Values {
				combined.Values[index] += task.Scale * (fineTuned.Values[index] - baseWeight.Values[index])
				if !checked.Finite32(combined.Values[index]) {
					return Result{}, errors.New("model merge: task arithmetic produced a non-finite weight")
				}
			}
			output[name] = combined
		}
		parents = append(parents, task.Model.ID)
	}
	model, err := compiler.Seal(base.Definition, base.ID, output)
	if err != nil {
		return Result{}, err
	}
	return Result{Model: model, Lineage: artifact.DependencyLineage(model.ID, parents...)}, nil
}

func (Compiler) identity(snapshot Snapshot) (artifact.ID, error) {
	if snapshot.Definition.Kind() != artifact.KindModelDefinition ||
		snapshot.Base.Valid() && snapshot.Base.Kind() != artifact.KindModel ||
		!checked.Nonzero(len(snapshot.Tensors)) {
		return artifact.ID{}, errors.New("model merge: snapshot model definition, base, or inventory is invalid")
	}
	for name, weight := range snapshot.Tensors {
		if name == "" {
			return artifact.ID{}, errors.New("model merge: tensor name is empty")
		}
		elements, err := weightElements(weight.Layout)
		if err != nil || elements != uint64(len(weight.Values)) {
			return artifact.ID{}, fmt.Errorf("model merge: tensor %q layout or storage differs", name)
		}
		for _, value := range weight.Values {
			if !checked.Finite32(value) {
				return artifact.ID{}, fmt.Errorf("model merge: tensor %q contains a non-finite value", name)
			}
		}
	}
	body := struct {
		Definition artifact.ID       `json:"definition"`
		Base       artifact.ID       `json:"base,omitzero"`
		Tensors    map[string]Weight `json:"tensors"`
	}{Definition: snapshot.Definition, Base: snapshot.Base, Tensors: snapshot.Tensors}
	return artifact.JSONID(artifact.KindModel, body)
}

func compatibleInventory(reference, candidate Snapshot) error {
	if reference.Definition != candidate.Definition || len(reference.Tensors) != len(candidate.Tensors) {
		return errors.New("model merge: model definition or tensor inventory differs")
	}
	for name, expected := range reference.Tensors {
		actual, ok := candidate.Tensors[name]
		if !ok || expected.Layout != actual.Layout || len(expected.Values) != len(actual.Values) {
			return fmt.Errorf("model merge: tensor %q geometry differs", name)
		}
	}
	return nil
}

func cloneWeights(weights map[string]Weight) map[string]Weight {
	cloned := make(map[string]Weight, len(weights))
	names := make([]string, tensor.FirstOffset, len(weights))
	for name := range weights {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		cloned[name] = cloneWeight(weights[name])
	}
	return cloned
}

func cloneWeight(weight Weight) Weight {
	weight.Values = slices.Clone(weight.Values)
	return weight
}
