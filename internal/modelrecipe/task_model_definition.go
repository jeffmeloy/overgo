package modelrecipe

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
)

// A task definition binds existing operation and tensor authorities without
// fabricating a language-model Spec for an independently declared executor.
type taskModelDefinition struct {
	Version uint16      `json:"version"`
	Model   artifact.ID `json:"model"`
	Recipe  artifact.ID `json:"recipe"`
	ID      artifact.ID `json:"-"`
}

var taskModelDefinitionCodec = artifact.JSONDocumentCodec(
	"task model definition", artifact.KindModelDefinition,
	"application/vnd.overgo.task-model-definition+json", "overgo/task-model-definition/v1",
	func(value *taskModelDefinition) error {
		if value.Version != artifact.InitialDocumentVersion || value.Model.Kind() != artifact.KindModel || value.Recipe.Kind() != artifact.KindRecipe {
			return errors.New("model recipe: invalid task model definition")
		}
		return nil
	},
	func(value taskModelDefinition) artifact.ID { return value.ID },
	func(value *taskModelDefinition, id artifact.ID) { value.ID = id }, nil,
)

func taskModelDefinitionLineage(value taskModelDefinition) []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Model, value.Recipe)
}

// PublishTaskModelDefinition binds an existing executable recipe, model manifest
// and tensor inventory. It neither activates the recipe nor executes a model.
func PublishTaskModelDefinition(ctx context.Context, repository artifact.Repository, recipeID artifact.ID) (artifact.ID, error) {
	if ctx == nil || repository == nil || recipeID.Kind() != artifact.KindRecipe {
		return artifact.ID{}, errors.New("model recipe: invalid task definition publication")
	}
	definition, err := recipe.RequireDefinition(ctx, repository, recipeID)
	if err != nil {
		return artifact.ID{}, err
	}
	value, err := taskModelDefinitionCodec.NewInitial(taskModelDefinition{Model: definition.Model, Recipe: definition.ID})
	if err != nil {
		return artifact.ID{}, err
	}
	if err := requireTaskModelDefinition(ctx, repository, value); err != nil {
		return artifact.ID{}, err
	}
	batch, err := taskModelDefinitionCodec.Batch("recipe/task-definition/"+value.ID.String(), value, taskModelDefinitionLineage(value), nil)
	if err != nil {
		return artifact.ID{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	if errors.Is(err, artifact.ErrNoChange) {
		err = nil
	}
	return value.ID, err
}

// RequireModelDefinitionBinding validates either resolved architecture metadata
// or an exact operation-defined model. Task definitions must name the executed
// recipe; descriptor-only, foreign-model and unknown-schema claims are refused.
func RequireModelDefinitionBinding(ctx context.Context, reader artifact.Reader, definitionID, modelID, runtimeRecipeID artifact.ID) error {
	if ctx == nil || reader == nil || definitionID.Kind() != artifact.KindModelDefinition || modelID.Kind() != artifact.KindModel || runtimeRecipeID.Kind() != artifact.KindRecipe {
		return errors.New("model recipe: invalid model definition binding")
	}
	descriptor, found, err := reader.Artifact(ctx, definitionID)
	if err != nil || !found {
		return errors.Join(err, errors.New("model recipe: model definition is absent"))
	}
	if descriptor.Schema == ModelDefinitionSchema {
		definition, err := ResolveModelDefinition(ctx, reader, definitionID)
		if err != nil || definition.Document.Model != modelID {
			return errors.Join(err, errors.New("model recipe: model definition differs from model"))
		}
		return nil
	}
	value, err := taskModelDefinitionCodec.RequireExactLineage(ctx, reader, definitionID, taskModelDefinitionLineage)
	if err != nil || value.Model != modelID || value.Recipe != runtimeRecipeID {
		return errors.Join(err, errors.New("model recipe: task definition differs from execution"))
	}
	return requireTaskModelDefinition(ctx, reader, value)
}

func requireTaskModelDefinition(ctx context.Context, reader artifact.Reader, value taskModelDefinition) error {
	definition, err := recipe.RequireDefinition(ctx, reader, value.Recipe)
	if err != nil || definition.Model != value.Model {
		return errors.Join(err, errors.New("model recipe: task definition model differs"))
	}
	if _, err := CompileCapability(definition); err != nil {
		return err
	}
	manifest, found, err := reader.Manifest(ctx, value.Model)
	if err != nil || !found || manifest.Validate() != nil {
		return errors.Join(err, errors.New("model recipe: task model manifest is absent or invalid"))
	}
	inventoryID, found := definition.PrimaryDependency(recipe.DependencyTensorInventory)
	if !found {
		return errors.New("model recipe: task definition requires a tensor inventory")
	}
	inventory, found, err := modelartifact.ReadTensorInventoryDocument(ctx, reader, inventoryID)
	if err != nil || !found || inventory.Owner != value.Model {
		return errors.Join(err, errors.New("model recipe: task tensor inventory differs"))
	}
	return nil
}
