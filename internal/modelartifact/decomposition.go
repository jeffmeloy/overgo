package modelartifact

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/organ"
)

const (
	ComponentDecompositionVersion   uint16 = 1
	ComponentDecompositionMediaType        = "application/vnd.overgo.component-decomposition+json"
	ComponentDecompositionSchema           = "overgo/component-decomposition/v1"
)

// ComponentContract is one tensor's typed component classification: the
// tensor's identity facts plus its eight-axis organ contract.
type ComponentContract struct {
	Name     string         `json:"name"`
	Shape    []uint64       `json:"shape,omitempty"`
	Storage  string         `json:"storage,omitzero"`
	Contract organ.Contract `json:"contract"`
}

// ComponentDecompositionDocument is a model's classification-only component
// decomposition: every tensor typed across the organ axes, lineage-bound to
// the source model. No retrieval index -- the catalog row builds pools on top
// of these documents.
type ComponentDecompositionDocument struct {
	Version    uint16              `json:"version"`
	Model      artifact.ID         `json:"model"`
	Family     string              `json:"family,omitzero"`
	Modality   string              `json:"modality,omitzero"`
	Components []ComponentContract `json:"components"`
	ID         artifact.ID         `json:"-"`
}

var componentDecompositionCodec = artifact.JSONDocumentCodec(
	"model artifact component decomposition", artifact.KindTensorSet,
	ComponentDecompositionMediaType, ComponentDecompositionSchema,
	canonicalizeComponentDecomposition,
	func(value ComponentDecompositionDocument) artifact.ID { return value.ID },
	func(value *ComponentDecompositionDocument, id artifact.ID) { value.ID = id },
	func(value ComponentDecompositionDocument) ComponentDecompositionDocument {
		value.Components = slices.Clone(value.Components)
		return value
	},
)

// NewComponentDecomposition classifies every tensor fact across the organ
// axes and identifies the resulting document. Classification is deterministic
// in the inputs, so re-decomposing an unchanged model reproduces the
// identical artifact.
func NewComponentDecomposition(
	model artifact.ID,
	family, modality string,
	tensors []TensorFact,
) (ComponentDecompositionDocument, error) {
	if len(tensors) == 0 {
		return ComponentDecompositionDocument{}, errors.New("model artifact: decomposition requires tensor facts")
	}
	components := make([]ComponentContract, 0, len(tensors))
	for _, fact := range tensors {
		if strings.TrimSpace(fact.Name) == "" {
			return ComponentDecompositionDocument{}, errors.New("model artifact: decomposition tensor lacks a name")
		}
		components = append(components, ComponentContract{
			Name: fact.Name, Shape: slices.Clone(fact.Shape), Storage: fact.Storage,
			Contract: organ.Classify(fact.Name, fact.Storage, family, modality, ""),
		})
	}
	slices.SortFunc(components, func(left, right ComponentContract) int {
		return strings.Compare(left.Name, right.Name)
	})
	return componentDecompositionCodec.New(ComponentDecompositionDocument{
		Version: ComponentDecompositionVersion, Model: model,
		Family: family, Modality: modality, Components: components,
	})
}

func ParseComponentDecomposition(content []byte) (ComponentDecompositionDocument, error) {
	return componentDecompositionCodec.Parse(content)
}

func (d ComponentDecompositionDocument) Content() (artifact.Content, error) {
	return componentDecompositionCodec.Content(d)
}

func (d ComponentDecompositionDocument) Lineage() artifact.Lineage {
	return artifact.Lineage{Child: d.ID, Parent: d.Model, Relation: artifact.RelationDerivedFrom}
}

func (d ComponentDecompositionDocument) Batch(key string) (artifact.Batch, error) {
	return componentDecompositionCodec.Batch(key, d, []artifact.Lineage{d.Lineage()}, nil)
}

func canonicalizeComponentDecomposition(value *ComponentDecompositionDocument) error {
	if value == nil || value.Version != ComponentDecompositionVersion {
		return errors.New("model artifact: invalid component decomposition version")
	}
	if value.Model.Kind() != artifact.KindModel {
		return errors.New("model artifact: decomposition source must be a model artifact")
	}
	if len(value.Components) == 0 {
		return errors.New("model artifact: decomposition carries no components")
	}
	for index, component := range value.Components {
		if strings.TrimSpace(component.Name) == "" {
			return errors.New("model artifact: decomposition component lacks a name")
		}
		if index > 0 && value.Components[index-1].Name >= component.Name {
			return errors.New("model artifact: decomposition components must be strictly name-ordered")
		}
		if issues := organ.Validate(component.Contract); len(issues) != 0 {
			return fmt.Errorf("model artifact: component %s contract invalid: %s", component.Name, issues[0].Message)
		}
	}
	return nil
}
