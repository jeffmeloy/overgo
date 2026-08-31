package modelrecipe

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/model"
)

const (
	// ModelPrototypeMediaType identifies one non-authorizing model-family prototype.
	ModelPrototypeMediaType = "application/vnd.overgo.model-prototype+json"
	// ModelPrototypeSchema identifies the model prototype contract version.
	ModelPrototypeSchema = "overgo/model-prototype/v1"
)

type modelPrototypeDocument struct {
	Version      uint16      `json:"version"`
	Architecture string      `json:"architecture"`
	Profile      artifact.ID `json:"profile"`
	ID           artifact.ID `json:"-"`
}

// ModelPrototype identifies existing compiled Go architecture behavior through
// its exact registered profile. The profile already owns tensor namespaces,
// operators, routing, numerical policy, and execution modes; this document
// deliberately does not copy them or bind weights, activation, or capability.
type ModelPrototype struct {
	document modelPrototypeDocument
}

var modelPrototypeCodec = artifact.JSONDocumentCodec(
	"model prototype", artifact.KindRecipe, ModelPrototypeMediaType, ModelPrototypeSchema,
	canonicalizeModelPrototype,
	func(value modelPrototypeDocument) artifact.ID { return value.ID },
	func(value *modelPrototypeDocument, id artifact.ID) { value.ID = id }, nil,
)

// NewModelPrototype identifies one exact registered architecture profile as a prototype.
func NewModelPrototype(profile ProfileDocument) (ModelPrototype, error) {
	if err := profile.ValidateIdentity(); err != nil {
		return ModelPrototype{}, err
	}
	document, err := modelPrototypeCodec.New(modelPrototypeDocument{
		Version: artifact.InitialDocumentVersion, Architecture: profile.Architecture, Profile: profile.ID,
	})
	return ModelPrototype{document: document}, err
}

// RequireModelPrototype loads one exact prototype and closes it against the
// currently compiled and published architecture profile.
func RequireModelPrototype(ctx context.Context, reader artifact.Reader, id artifact.ID) (ModelPrototype, error) {
	document, err := modelPrototypeCodec.Require(ctx, reader, id)
	if err != nil {
		return ModelPrototype{}, err
	}
	value := ModelPrototype{document: document}
	profile, err := ResolveRegisteredArchitectureProfile(ctx, reader, value.Architecture())
	if err != nil {
		return ModelPrototype{}, err
	}
	canonical, err := NewModelPrototype(profile)
	if err != nil || canonical.ID() != value.ID() || profile.ID != value.Profile() {
		return ModelPrototype{}, errors.Join(err, errors.New("model recipe: stored prototype differs from its compiled profile"))
	}
	parents, err := reader.Parents(ctx, value.ID())
	if err != nil {
		return ModelPrototype{}, err
	}
	if !slices.Contains(parents, artifact.Lineage{
		Child: value.ID(), Parent: value.Profile(), Relation: artifact.RelationDependsOn,
	}) {
		return ModelPrototype{}, errors.New("model recipe: prototype profile lineage is absent")
	}
	return value, nil
}

// ID returns the prototype's content identity.
func (value ModelPrototype) ID() artifact.ID { return value.document.ID }

// Architecture returns the exact compiled registry name.
func (value ModelPrototype) Architecture() string { return value.document.Architecture }

// Profile returns the exact compiled architecture profile identity.
func (value ModelPrototype) Profile() artifact.ID { return value.document.Profile }

// ValidateIdentity verifies the prototype's canonical content identity.
func (value ModelPrototype) ValidateIdentity() error {
	return modelPrototypeCodec.ValidateIdentity(value.document)
}

// Content returns the canonical model prototype document.
func (value ModelPrototype) Content() (artifact.Content, error) {
	return modelPrototypeCodec.Content(value.document)
}

// Lineage binds the prototype to its exact registered architecture profile.
func (value ModelPrototype) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.document.ID, value.document.Profile)
}

func canonicalizeModelPrototype(value *modelPrototypeDocument) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Profile.Kind() != artifact.KindProfile || model.ValidateArchitectureName(value.Architecture) != nil {
		return errors.New("model recipe: invalid model prototype")
	}
	registered, found := model.LookupArchitecture(value.Architecture)
	if !found {
		return errors.New("model recipe: model prototype architecture is not compiled into Go")
	}
	profile, err := NewProfileDocument(registered)
	if err != nil {
		return err
	}
	if profile.ID != value.Profile {
		return errors.New("model recipe: model prototype profile differs from the compiled Go registration")
	}
	return nil
}
