package trainingprogram

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
)

const (
	ObjectiveCompositionMediaType = "application/vnd.overgo.objective-composition+json"
	ObjectiveCompositionSchema    = "overgo/objective-composition/v1"
)

type ObjectiveCompositionSpec struct {
	ObjectiveSpec
	Components []artifact.ID `json:"components"`
}

// ObjectiveComposition binds compatible objectives to one aggregate corpus.
type ObjectiveComposition struct {
	ID      artifact.ID `json:"-"`
	Version uint16      `json:"version"`
	ObjectiveCompositionSpec
}

var objectiveCompositionCodec = artifact.JSONDocumentCodec(
	"objective composition", artifact.KindProfile, ObjectiveCompositionMediaType, ObjectiveCompositionSchema,
	canonicalizeObjectiveComposition,
	func(value ObjectiveComposition) artifact.ID { return value.ID },
	func(value *ObjectiveComposition, id artifact.ID) { value.ID = id },
	func(value ObjectiveComposition) ObjectiveComposition {
		value.Signature = value.Signature.Clone()
		value.Processors = slices.Clone(value.Processors)
		value.Projectors = slices.Clone(value.Projectors)
		value.Codecs = slices.Clone(value.Codecs)
		value.Evidence = slices.Clone(value.Evidence)
		value.Components = slices.Clone(value.Components)
		return value
	},
)

func NewObjectiveComposition(spec ObjectiveCompositionSpec) (ObjectiveComposition, error) {
	return objectiveCompositionCodec.New(ObjectiveComposition{
		Version: artifact.InitialDocumentVersion, ObjectiveCompositionSpec: spec,
	})
}

func (c ObjectiveComposition) Content() (artifact.Content, error) {
	return objectiveCompositionCodec.Content(c)
}

func (c ObjectiveComposition) Lineage() []artifact.Lineage {
	result := make([]artifact.Lineage, len(c.Components))
	for index, component := range c.Components {
		result[index] = artifact.Lineage{Child: c.ID, Parent: component, Relation: artifact.RelationDerivedFrom}
	}
	return result
}

func canonicalizeObjectiveComposition(value *ObjectiveComposition) error {
	if value.Version != artifact.InitialDocumentVersion {
		return errors.New("objective composition: unsupported version")
	}
	leaf := ObjectiveDocument{Version: artifact.SecondDocumentVersion, ObjectiveSpec: value.ObjectiveSpec}
	if err := canonicalizeObjective(&leaf); err != nil {
		return fmt.Errorf("objective composition: %w", err)
	}
	value.ObjectiveSpec = leaf.ObjectiveSpec
	value.Components = slices.Clone(value.Components)
	sort.Slice(value.Components, func(left, right int) bool {
		return value.Components[left].String() < value.Components[right].String()
	})
	if len(value.Components) < 2 {
		return errors.New("objective composition: at least two components required")
	}
	for index, component := range value.Components {
		if component.Kind() != artifact.KindProfile || index > 0 && value.Components[index-1] == component {
			return errors.New("objective composition: invalid or duplicate component")
		}
	}
	return nil
}

type resolvedObjective struct {
	spec ObjectiveSpec
}

func resolveObjective(ctx context.Context, reader artifact.Reader, id artifact.ID) (resolvedObjective, error) {
	if ctx == nil || reader == nil || id.Kind() != artifact.KindProfile {
		return resolvedObjective{}, errors.New("training objective: invalid repository authority")
	}
	content, ok, err := reader.Content(ctx, id)
	if err != nil {
		return resolvedObjective{}, err
	}
	if !ok {
		return resolvedObjective{}, errors.New("training objective: RepoDB content absent")
	}
	switch content.Descriptor.MediaType {
	case ObjectiveMediaType:
		document, err := LoadObjective(ctx, reader, id)
		if err != nil {
			return resolvedObjective{}, err
		}
		if err := validateObjectiveReferences(ctx, reader, document); err != nil {
			return resolvedObjective{}, err
		}
		return resolvedObjective{spec: document.ObjectiveSpec}, nil
	case ObjectiveCompositionMediaType:
		composition, ok, err := objectiveCompositionCodec.Read(ctx, reader, id)
		if err != nil || !ok {
			return resolvedObjective{}, errors.Join(err, errors.New("objective composition: RepoDB content absent"))
		}
		return resolveObjectiveComposition(ctx, reader, composition)
	default:
		return resolvedObjective{}, fmt.Errorf("training objective: unsupported media type %q", content.Descriptor.MediaType)
	}
}

func resolveObjectiveComposition(ctx context.Context, reader artifact.Reader, composition ObjectiveComposition) (resolvedObjective, error) {
	aggregate := ObjectiveDocument{ID: composition.ID, Version: artifact.SecondDocumentVersion, ObjectiveSpec: composition.ObjectiveSpec}
	if err := validateObjectiveReferences(ctx, reader, aggregate); err != nil {
		return resolvedObjective{}, err
	}
	for _, id := range composition.Components {
		component, err := LoadObjective(ctx, reader, id)
		if err != nil {
			return resolvedObjective{}, fmt.Errorf("objective composition: component %s: %w", id, err)
		}
		if err := validateObjectiveReferences(ctx, reader, component); err != nil {
			return resolvedObjective{}, fmt.Errorf("objective composition: component %s: %w", id, err)
		}
		if !compatibleObjective(component.ObjectiveSpec, composition.ObjectiveSpec) {
			return resolvedObjective{}, fmt.Errorf("objective composition: component %s contract differs", id)
		}
	}
	return resolvedObjective{spec: composition.ObjectiveSpec}, nil
}

func compatibleObjective(component, aggregate ObjectiveSpec) bool {
	return component.Kind == aggregate.Kind && component.Loss == aggregate.Loss &&
		component.Evaluation == aggregate.Evaluation && component.Metric == aggregate.Metric &&
		component.Authority == aggregate.Authority &&
		slices.Equal(component.Signature.Inputs, aggregate.Signature.Inputs) &&
		slices.Equal(component.Signature.Outputs, aggregate.Signature.Outputs) &&
		slices.Equal(component.Processors, aggregate.Processors) &&
		slices.Equal(component.Projectors, aggregate.Projectors) &&
		slices.Equal(component.Codecs, aggregate.Codecs)
}
