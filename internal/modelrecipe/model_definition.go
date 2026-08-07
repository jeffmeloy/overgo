package modelrecipe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
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

type modelDefinitionBody struct {
	Version         uint16      `json:"version"`
	Model           artifact.ID `json:"model"`
	Profile         artifact.ID `json:"profile"`
	TensorInventory artifact.ID `json:"tensor_inventory"`
	Architecture    string      `json:"architecture"`
	Spec            model.Spec  `json:"spec"`
}

// ModelDefinitionDocument: resolved model metadata bindings.
type ModelDefinitionDocument struct {
	ID              artifact.ID
	Version         uint16
	Model           artifact.ID
	Profile         artifact.ID
	TensorInventory artifact.ID
	Architecture    string
	Spec            model.Spec
}

// ResolvedModelDefinition: definition plus exact policy and tensor facts.
type ResolvedModelDefinition struct {
	Document ModelDefinitionDocument
	Profile  ProfileDocument
	Tensors  modelartifact.TensorInventoryDocument
	Spec     model.Spec
}

// Batch: atomic model inventory and definition publication.
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

// PublishResolvedModelDefinition: atomic bound-definition commit.
func PublishResolvedModelDefinition(
	ctx context.Context,
	store artifact.Repository,
	key string,
	inventory modelartifact.Inventory,
	resolved ResolvedModelDefinition,
) (artifact.CommitID, error) {
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
		Version: ModelDefinitionVersion, Model: tensors.Model,
		Profile: profile.ID, TensorInventory: tensors.ID,
		Architecture: profile.Architecture, Spec: bound,
	}
	content, err := modelDefinitionContent(document)
	if err != nil {
		return ModelDefinitionDocument{}, err
	}
	if len(content) > artifact.MaxContentBytes {
		return ModelDefinitionDocument{}, errors.New("model recipe: model definition exceeds inline content limit")
	}
	var cloned modelDefinitionBody
	if err := strictjson.DecodeBytes(content, &cloned); err != nil {
		return ModelDefinitionDocument{}, err
	}
	document.Spec = cloned.Spec
	document.ID, err = modelDefinitionContract.Identify(content)
	return document, err
}

func NewModelDefinitionFromGGUF(
	file *gguf.File,
	profile ProfileDocument,
	tensors modelartifact.TensorInventoryDocument,
) (ModelDefinitionDocument, error) {
	if file == nil {
		return ModelDefinitionDocument{}, errors.New("model recipe: nil GGUF model definition source")
	}
	observed, err := modelartifact.NewGGUFTensorInventory(tensors.Model, file)
	if err != nil {
		return ModelDefinitionDocument{}, err
	}
	if observed.ID != tensors.ID {
		return ModelDefinitionDocument{}, errors.New("model recipe: GGUF tensor inventory does not match definition source")
	}
	spec, err := model.ReadSpecWithProfile(file, profile.Policy)
	if err != nil {
		return ModelDefinitionDocument{}, err
	}
	return NewModelDefinitionDocument(profile, tensors, spec)
}

func ParseModelDefinitionDocument(content []byte) (ModelDefinitionDocument, error) {
	var body modelDefinitionBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return ModelDefinitionDocument{}, fmt.Errorf("model recipe: decode model definition: %w", err)
	}
	document := ModelDefinitionDocument{
		Version: body.Version, Model: body.Model, Profile: body.Profile,
		TensorInventory: body.TensorInventory, Architecture: body.Architecture, Spec: body.Spec,
	}
	if err := document.validateShape(); err != nil {
		return ModelDefinitionDocument{}, err
	}
	canonical, err := modelDefinitionContent(document)
	if err != nil {
		return ModelDefinitionDocument{}, err
	}
	if !bytes.Equal(content, canonical) {
		return ModelDefinitionDocument{}, errors.New("model recipe: non-canonical model definition")
	}
	document.ID, err = modelDefinitionContract.Identify(content)
	return document, err
}

func (d ModelDefinitionDocument) ValidateIdentity() error {
	if d.ID.Kind() != artifact.KindModelDefinition {
		return errors.New("model recipe: invalid model definition identity")
	}
	if err := d.validateShape(); err != nil {
		return err
	}
	content, err := modelDefinitionContent(d)
	if err != nil {
		return err
	}
	if err := modelDefinitionContract.ValidateIdentity(d.ID, content); err != nil {
		return errors.New("model recipe: model definition identity mismatch")
	}
	return nil
}

func (d ModelDefinitionDocument) ContentBytes() ([]byte, error) {
	if err := d.ValidateIdentity(); err != nil {
		return nil, err
	}
	return modelDefinitionContent(d)
}

func (d ModelDefinitionDocument) Content() (artifact.Content, error) {
	content, err := d.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return modelDefinitionContract.Content(d.ID, content)
}

func (d ModelDefinitionDocument) Batch(key string) (artifact.Batch, error) {
	content, err := d.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, []artifact.Lineage{
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
	if d.Model != tensors.Model || d.Profile != profile.ID || d.TensorInventory != tensors.ID ||
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
	content, ok, err := artifact.ReadDocument(ctx, store, id, modelDefinitionContract)
	if err != nil {
		return ResolvedModelDefinition{}, err
	}
	if !ok {
		return ResolvedModelDefinition{}, errors.New("model recipe: model definition content is absent or incompatible")
	}
	document, err := ParseModelDefinitionDocument(content.Data)
	if err != nil {
		return ResolvedModelDefinition{}, err
	}
	if document.ID != id {
		return ResolvedModelDefinition{}, errors.New("model recipe: resolved model definition identity mismatch")
	}
	profile, err := loadProfile(ctx, store, document.Profile)
	if err != nil {
		return ResolvedModelDefinition{}, err
	}
	tensorContent, ok, err := artifact.ReadDocument(
		ctx, store, document.TensorInventory, modelartifact.TensorInventoryDocumentContract(),
	)
	if err != nil {
		return ResolvedModelDefinition{}, err
	}
	if !ok {
		return ResolvedModelDefinition{}, errors.New("model recipe: tensor inventory content is absent or incompatible")
	}
	tensors, err := modelartifact.ParseTensorInventoryDocument(tensorContent.Data)
	if err != nil {
		return ResolvedModelDefinition{}, err
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
	if definition.Version == recipe.LegacyVersion {
		return Plan{}, errors.New("model recipe: stored model definition requires recipe v2")
	}
	definitionID, ok := definition.Dependency(recipe.DependencyDefinition, 0)
	if !ok || definitionID != resolved.Document.ID || definition.Model != resolved.Document.Model {
		return Plan{}, errors.New("model recipe: recipe model definition binding mismatch")
	}
	profileID, ok := definition.Dependency(recipe.DependencyProfile, 0)
	if !ok || profileID != resolved.Profile.ID {
		return Plan{}, errors.New("model recipe: recipe profile binding mismatch")
	}
	return compileDefinition(definition, func() (model.ModelPlan, error) {
		return model.CompileModelPlanWithProfile(resolved.Spec, weights, resolved.Profile.Policy)
	})
}

func (d ModelDefinitionDocument) validateShape() error {
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
	return content, nil
}
