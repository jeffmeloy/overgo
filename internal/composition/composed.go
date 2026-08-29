package composition

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/tensor"
)

// OfflineCompositionOperation identifies an immutable, non-serving model
// artifact workflow.
type OfflineCompositionOperation string

const (
	// OfflineCompositionPassthrough preserves one base model behind an exact
	// compatibility authority without mutating its payload.
	OfflineCompositionPassthrough OfflineCompositionOperation = "passthrough"
	// OfflineCompositionTaskArithmetic applies exact rational coefficients to
	// independently cataloged task-vector tensor sets.
	OfflineCompositionTaskArithmetic OfflineCompositionOperation = "task-arithmetic"
)

// TaskArithmeticTerm binds one tensor-set delta to an exact rational scalar.
type TaskArithmeticTerm struct {
	Component   artifact.ID `json:"component"`
	Numerator   int64       `json:"numerator"`
	Denominator uint64      `json:"denominator"`
}

const (
	ComposedModelVersion   uint16 = 1
	ComposedModelMediaType        = "application/vnd.overgo.composed-model+json"
	ComposedModelSchema           = "overgo/composed-model/v1"
)

// ComposedModelDocument: executable composition and immutable inputs.
type ComposedModelDocument struct {
	Version       uint16                      `json:"version"`
	Architecture  string                      `json:"architecture"`
	Recipe        artifact.ID                 `json:"recipe"`
	Parents       []artifact.ID               `json:"parents"`
	Components    []artifact.ID               `json:"components,omitempty"`
	Adapter       artifact.ID                 `json:"adapter,omitzero"`
	Checkpoint    artifact.ID                 `json:"checkpoint,omitzero"`
	Operation     OfflineCompositionOperation `json:"operation,omitzero"`
	Compatibility []artifact.ID               `json:"compatibility,omitempty"`
	Arithmetic    []TaskArithmeticTerm        `json:"arithmetic,omitempty"`
	ID            artifact.ID                 `json:"-"`
}

var composedModelCodec = artifact.JSONDocumentCodec(
	"composed model", artifact.KindModel, ComposedModelMediaType, ComposedModelSchema,
	canonicalizeComposedModel,
	func(value ComposedModelDocument) artifact.ID { return value.ID },
	func(value *ComposedModelDocument, id artifact.ID) { value.ID = id },
	func(value ComposedModelDocument) ComposedModelDocument {
		value.Parents = slices.Clone(value.Parents)
		value.Components = slices.Clone(value.Components)
		value.Compatibility = slices.Clone(value.Compatibility)
		value.Arithmetic = slices.Clone(value.Arithmetic)
		return value
	},
)

// NewComposedModel identifies one composed artifact.
func NewComposedModel(value ComposedModelDocument) (ComposedModelDocument, error) {
	value.Version = ComposedModelVersion
	value.ID = artifact.ID{}
	return composedModelCodec.New(value)
}

func ParseComposedModel(content []byte) (ComposedModelDocument, error) {
	return composedModelCodec.Parse(content)
}

func (d ComposedModelDocument) Content() (artifact.Content, error) {
	return composedModelCodec.Content(d)
}

func (d ComposedModelDocument) Lineage() []artifact.Lineage {
	capacity := 1 + len(d.Parents) + len(d.Components) + len(d.Compatibility)
	if d.Adapter.Valid() {
		capacity++
	}
	if d.Checkpoint.Valid() {
		capacity++
	}
	dependencies := make([]artifact.ID, 0, capacity)
	dependencies = append(dependencies, d.Recipe)
	dependencies = append(dependencies, d.Parents...)
	dependencies = append(dependencies, d.Components...)
	dependencies = append(dependencies, d.Compatibility...)
	if d.Adapter.Valid() {
		dependencies = append(dependencies, d.Adapter)
	}
	if d.Checkpoint.Valid() {
		dependencies = append(dependencies, d.Checkpoint)
	}
	return artifact.DependencyLineage(d.ID, dependencies...)
}

func (d ComposedModelDocument) Batch(key string) (artifact.Batch, error) {
	return composedModelCodec.Batch(key, d, d.Lineage(), nil)
}

func canonicalizeComposedModel(value *ComposedModelDocument) error {
	if value == nil || value.Version != ComposedModelVersion {
		return errors.New("composition: invalid composed model version")
	}
	if strings.TrimSpace(value.Architecture) == "" {
		return errors.New("composition: composed model requires an architecture")
	}
	if _, registered := model.LookupArchitecture(value.Architecture); !registered {
		return fmt.Errorf("composition: architecture %q is not in the executor vocabulary", value.Architecture)
	}
	if value.Recipe.Kind() != artifact.KindRecipe {
		return errors.New("composition: composed model requires its executable recipe")
	}
	if len(value.Parents) == 0 {
		return errors.New("composition: composed model requires parent models")
	}
	for _, parent := range value.Parents {
		if parent.Kind() != artifact.KindModel {
			return errors.New("composition: composed parent is not a model")
		}
	}
	for _, component := range value.Components {
		if component.Kind() != artifact.KindTensorSet {
			return errors.New("composition: composed component is not a tensor set")
		}
	}
	if value.Adapter.Valid() && value.Adapter.Kind() != artifact.KindAdapter {
		return errors.New("composition: composed adapter reference kind mismatch")
	}
	if value.Checkpoint.Valid() && value.Checkpoint.Kind() != artifact.KindCheckpoint {
		return errors.New("composition: composed checkpoint reference kind mismatch")
	}
	if err := validateOfflineComposition(value); err != nil {
		return err
	}
	return nil
}

func validateOfflineComposition(value *ComposedModelDocument) error {
	if value.Operation == "" {
		if len(value.Compatibility) != 0 || len(value.Arithmetic) != 0 {
			return errors.New("composition: offline authorities require an operation")
		}
		return nil
	}
	if len(value.Parents) != tensor.SingletonExtent || len(value.Compatibility) == 0 {
		return errors.New("composition: offline composition requires one base and compatibility authority")
	}
	seenCompatibility := make(map[artifact.ID]struct{}, len(value.Compatibility))
	for _, id := range value.Compatibility {
		if id.Kind() != artifact.KindProfile && id.Kind() != artifact.KindTensorInventory {
			return errors.New("composition: offline compatibility authority kind mismatch")
		}
		if _, duplicate := seenCompatibility[id]; duplicate {
			return errors.New("composition: offline compatibility authority is duplicated")
		}
		seenCompatibility[id] = struct{}{}
	}
	switch value.Operation {
	case OfflineCompositionPassthrough:
		if len(value.Components) != 0 || len(value.Arithmetic) != 0 ||
			value.Adapter.Valid() || value.Checkpoint.Valid() {
			return errors.New("composition: passthrough cannot mutate the base model")
		}
	case OfflineCompositionTaskArithmetic:
		if len(value.Components) == 0 || len(value.Arithmetic) != len(value.Components) {
			return errors.New("composition: task arithmetic requires one coefficient per component")
		}
		seenComponents := make(map[artifact.ID]struct{}, len(value.Components))
		for index, term := range value.Arithmetic {
			if !checked.Equal(term.Component, value.Components[index]) ||
				term.Component.Kind() != artifact.KindTensorSet || term.Numerator == 0 ||
				!checked.Nonzero(term.Denominator) {
				return errors.New("composition: task arithmetic term differs from component authority")
			}
			if _, duplicate := seenComponents[term.Component]; duplicate {
				return errors.New("composition: task arithmetic component is duplicated")
			}
			seenComponents[term.Component] = struct{}{}
		}
	default:
		return errors.New("composition: offline composition operation is invalid")
	}
	return nil
}
