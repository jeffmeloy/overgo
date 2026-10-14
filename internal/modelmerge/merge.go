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

// Compiler has no runtime graph authority; its zero value seals
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
