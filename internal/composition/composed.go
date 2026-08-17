package composition

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/model"
)

const (
	ComposedModelVersion   uint16 = 1
	ComposedModelMediaType        = "application/vnd.overgo.composed-model+json"
	ComposedModelSchema           = "overgo/composed-model/v1"
)

// ComposedModelDocument assembles a composition's constituents into ONE
// content-addressed, executable model artifact: the parent models, the typed
// recipe that executes them, the classified components consumed, the
// registered architecture vocabulary, and optionally the trained adapter and
// the checkpoint a retry resumes from. Lineage binds the artifact to every
// input, so provenance is a graph query, never prose.
type ComposedModelDocument struct {
	Version      uint16        `json:"version"`
	Architecture string        `json:"architecture"`
	Recipe       artifact.ID   `json:"recipe"`
	Parents      []artifact.ID `json:"parents"`
	Components   []artifact.ID `json:"components,omitempty"`
	Adapter      artifact.ID   `json:"adapter,omitzero"`
	Checkpoint   artifact.ID   `json:"checkpoint,omitzero"`
	ID           artifact.ID   `json:"-"`
}

var composedModelCodec = artifact.JSONDocumentCodec(
	"composed model", artifact.KindModel, ComposedModelMediaType, ComposedModelSchema,
	canonicalizeComposedModel,
	func(value ComposedModelDocument) artifact.ID { return value.ID },
	func(value *ComposedModelDocument, id artifact.ID) { value.ID = id },
	func(value ComposedModelDocument) ComposedModelDocument {
		value.Parents = slices.Clone(value.Parents)
		value.Components = slices.Clone(value.Components)
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

// Batch commits the composed artifact with full lineage to every input.
func (d ComposedModelDocument) Batch(key string) (artifact.Batch, error) {
	content, err := d.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	dependencies := append([]artifact.ID{d.Recipe}, d.Parents...)
	dependencies = append(dependencies, d.Components...)
	if d.Adapter.Valid() {
		dependencies = append(dependencies, d.Adapter)
	}
	if d.Checkpoint.Valid() {
		dependencies = append(dependencies, d.Checkpoint)
	}
	lineage := make([]artifact.Lineage, 0, len(dependencies))
	for _, dependency := range dependencies {
		lineage = append(lineage, artifact.Lineage{
			Child: d.ID, Parent: dependency, Relation: artifact.RelationDependsOn,
		})
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, lineage, nil)
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
	return nil
}
