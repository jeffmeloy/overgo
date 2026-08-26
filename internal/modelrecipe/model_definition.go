package modelrecipe

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

const (
	ModelDefinitionVersion   uint16 = 1
	ModelDefinitionMediaType        = "application/vnd.overgo.model-definition+json"
	ModelDefinitionSchema           = "overgo/model-definition/v1"
)

var modelDefinitionContract = artifact.DocumentContract{
	Kind: artifact.KindModelDefinition, MediaType: ModelDefinitionMediaType, Schema: ModelDefinitionSchema,
}

var modelDefinitionCodec = artifact.DocumentCodec[ModelDefinitionDocument]{
	Name: "model definition", Contract: modelDefinitionContract,
	Decode: func(data []byte, value *ModelDefinitionDocument) error {
		var body modelDefinitionBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = ModelDefinitionDocument{
			Version: body.Version, Model: body.Model, Profile: body.Profile,
			TensorInventory: body.TensorInventory, Architecture: body.Architecture, Spec: body.Spec,
		}
		return nil
	},
	Encode: modelDefinitionContent, Canonicalize: canonicalizeModelDefinition,
	Identity:    func(value ModelDefinitionDocument) artifact.ID { return value.ID },
	SetIdentity: func(value *ModelDefinitionDocument, id artifact.ID) { value.ID = id },
}

type modelDefinitionBody struct {
	Version         uint16      `json:"version"`
	Model           artifact.ID `json:"model"`
	Profile         artifact.ID `json:"profile"`
	TensorInventory artifact.ID `json:"tensor_inventory"`
	Architecture    string      `json:"architecture"`
	Spec            model.Spec  `json:"spec"`
}

// ModelDefinitionDocument defines resolved model metadata bindings.
type ModelDefinitionDocument struct {
	ID              artifact.ID
	Version         uint16
	Model           artifact.ID
	Profile         artifact.ID
	TensorInventory artifact.ID
	Architecture    string
	Spec            model.Spec
}

// ResolvedModelDefinition defines definition plus exact policy and tensor facts.
type ResolvedModelDefinition struct {
	Document ModelDefinitionDocument
	Profile  ProfileDocument
	Tensors  modelartifact.TensorInventoryDocument
	Spec     model.Spec
}

// Batch returns an atomic model inventory and definition publication.
func (r ResolvedModelDefinition) Batch(
	key string,
	inventory modelartifact.Inventory,
) (artifact.Batch, error) {
	checked, err := r.Document.Resolve(r.Profile, r.Tensors)
	if err != nil {
		return artifact.Batch{}, err
	}
	if inventory.Manifest.ID != checked.Document.Model ||
		inventory.TensorInventory.ID != checked.Tensors.ID {
		return artifact.Batch{}, errors.New("model recipe: publication inventory binding mismatch")
	}
	batch, err := inventory.Batch(key)
	if err != nil {
		return artifact.Batch{}, err
	}
	profileContents, profileLineage, err := profilePublicationFacts(checked.Profile)
	if err != nil {
		return artifact.Batch{}, err
	}
	definitionContent, err := checked.Document.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	batch.Contents = append(batch.Contents, profileContents...)
	batch.Contents = append(batch.Contents, definitionContent)
	batch.Lineage = append(batch.Lineage, profileLineage...)
	batch.Lineage = append(batch.Lineage,
		artifact.Lineage{Child: checked.Document.ID, Parent: checked.Document.Model, Relation: artifact.RelationDerivedFrom},
		artifact.Lineage{Child: checked.Document.ID, Parent: checked.Profile.ID, Relation: artifact.RelationDependsOn},
		artifact.Lineage{Child: checked.Document.ID, Parent: checked.Tensors.ID, Relation: artifact.RelationDependsOn},
	)
	if err := batch.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
}

// PublishResolvedModelDefinition commits an atomic bound definition.
func PublishResolvedModelDefinition(
	ctx context.Context,
	store artifact.Repository,
	inventory modelartifact.Inventory,
	resolved ResolvedModelDefinition,
) (artifact.CommitID, error) {
	key := "recipe/facts/" + resolved.Document.ID.String()
	batch, err := resolved.Batch(key, inventory)
	if err != nil {
		return artifact.CommitID{}, err
	}
	return artifact.CommitBatch(ctx, store, batch)
}

func NewModelDefinitionDocument(
	profile ProfileDocument,
	tensors modelartifact.TensorInventoryDocument,
	spec model.Spec,
) (ModelDefinitionDocument, error) {
	if err := profile.ValidateIdentity(); err != nil {
		return ModelDefinitionDocument{}, err
	}
	if err := tensors.ValidateIdentity(); err != nil {
		return ModelDefinitionDocument{}, err
	}
	bound, err := model.BindSpecProfile(spec, profile.Policy)
	if err != nil {
		return ModelDefinitionDocument{}, err
	}
	document := ModelDefinitionDocument{
		Version: ModelDefinitionVersion, Model: tensors.Owner,
		Profile: profile.ID, TensorInventory: tensors.ID,
		Architecture: profile.Architecture, Spec: bound,
	}
	return modelDefinitionCodec.New(document)
}

func (d ModelDefinitionDocument) ValidateIdentity() error {
	return modelDefinitionCodec.ValidateIdentity(d)
}

func (d ModelDefinitionDocument) Content() (artifact.Content, error) {
	return modelDefinitionCodec.Content(d)
}

func (d ModelDefinitionDocument) Batch(key string) (artifact.Batch, error) {
	return modelDefinitionCodec.Batch(key, d, []artifact.Lineage{
		{Child: d.ID, Parent: d.Model, Relation: artifact.RelationDerivedFrom},
		{Child: d.ID, Parent: d.Profile, Relation: artifact.RelationDependsOn},
		{Child: d.ID, Parent: d.TensorInventory, Relation: artifact.RelationDependsOn},
	}, nil)
}

func (d ModelDefinitionDocument) Resolve(
	profile ProfileDocument,
	tensors modelartifact.TensorInventoryDocument,
) (ResolvedModelDefinition, error) {
	if err := d.ValidateIdentity(); err != nil {
		return ResolvedModelDefinition{}, err
	}
	if err := profile.ValidateIdentity(); err != nil {
		return ResolvedModelDefinition{}, err
	}
	if err := tensors.ValidateIdentity(); err != nil {
		return ResolvedModelDefinition{}, err
	}
	if d.Model != tensors.Owner || d.Profile != profile.ID || d.TensorInventory != tensors.ID ||
		d.Architecture != profile.Architecture || d.Spec.Architecture != d.Architecture {
		return ResolvedModelDefinition{}, errors.New("model recipe: model definition binding mismatch")
	}
	spec, err := model.BindSpecProfile(d.Spec, profile.Policy)
	if err != nil {
		return ResolvedModelDefinition{}, err
	}
	return ResolvedModelDefinition{Document: d, Profile: profile, Tensors: tensors, Spec: spec}, nil
}

func ResolveModelDefinition(
	ctx context.Context,
	store artifact.Reader,
	id artifact.ID,
) (ResolvedModelDefinition, error) {
	return resolveModelDefinition(ctx, store, id, nil)
}

func resolveModelDefinition(
	ctx context.Context,
	store artifact.Reader,
	id artifact.ID,
	boundProfile *ProfileDocument,
) (ResolvedModelDefinition, error) {
	document, err := modelDefinitionCodec.Require(ctx, store, id)
	if err != nil {
		return ResolvedModelDefinition{}, err
	}
	var profile ProfileDocument
	if boundProfile == nil {
		profile, err = loadProfile(ctx, store, document.Profile)
		if err != nil {
			return ResolvedModelDefinition{}, err
		}
	} else {
		profile = *boundProfile
		if profile.ID != document.Profile {
			return ResolvedModelDefinition{}, errors.New("model recipe: bound profile differs from model definition")
		}
	}
	tensors, ok, err := modelartifact.ReadTensorInventoryDocument(ctx, store, document.TensorInventory)
	if err != nil {
		return ResolvedModelDefinition{}, err
	}
	if !ok {
		return ResolvedModelDefinition{}, errors.New("model recipe: tensor inventory content is absent or incompatible")
	}
	return document.Resolve(profile, tensors)
}

func CompileModelDefinition(
	definition recipe.Definition,
	resolved ResolvedModelDefinition,
	weights model.Weights,
) (Plan, error) {
	checked, err := resolved.Document.Resolve(resolved.Profile, resolved.Tensors)
	if err != nil {
		return Plan{}, err
	}
	resolved = checked
	definitionID, ok := definition.PrimaryDependency(recipe.DependencyDefinition)
	if !ok || definitionID != resolved.Document.ID || definition.Model != resolved.Document.Model {
		return Plan{}, errors.New("model recipe: recipe model definition binding mismatch")
	}
	profileID, ok := definition.PrimaryDependency(recipe.DependencyProfile)
	if !ok || profileID != resolved.Profile.ID {
		return Plan{}, errors.New("model recipe: recipe profile binding mismatch")
	}
	return compileDefinition(definition, func() (model.ModelPlan, error) {
		return model.CompileModelPlanWithProfile(resolved.Spec, weights, resolved.Profile.Policy)
	})
}

func (d ModelDefinitionDocument) validate() error {
	if d.Version != ModelDefinitionVersion || d.Model.Kind() != artifact.KindModel ||
		d.Profile.Kind() != artifact.KindProfile || d.TensorInventory.Kind() != artifact.KindTensorInventory ||
		d.Architecture == "" || d.Spec.Architecture != d.Architecture {
		return errors.New("model recipe: invalid model definition envelope")
	}
	if err := model.ValidateArchitectureName(d.Architecture); err != nil {
		return fmt.Errorf("model recipe: invalid model definition architecture: %w", err)
	}
	return nil
}

func modelDefinitionContent(document ModelDefinitionDocument) ([]byte, error) {
	content, err := json.Marshal(modelDefinitionBody{
		Version: document.Version, Model: document.Model, Profile: document.Profile,
		TensorInventory: document.TensorInventory, Architecture: document.Architecture, Spec: document.Spec,
	})
	if err != nil {
		return nil, fmt.Errorf("model recipe: encode model definition: %w", err)
	}
	if len(content) > artifact.MaxContentBytes {
		return nil, errors.New("model recipe: model definition exceeds inline content limit")
	}
	return content, nil
}

func canonicalizeModelDefinition(document *ModelDefinitionDocument) error {
	if document == nil {
		return errors.New("model recipe: nil model definition")
	}
	if err := document.validate(); err != nil {
		return err
	}
	content, err := modelDefinitionContent(*document)
	if err != nil {
		return err
	}
	var body modelDefinitionBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return err
	}
	document.Spec = body.Spec
	return nil
}
