package modelrecipe

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// ModuleRemoteRelay is the inference module of a remote model: the prompt
// leaves for the declared provider's chat completions endpoint and the
// text comes back. No model plan compiles here, so the runner never opens
// such a recipe, while the servable catalog lists the model beside local
// activations and the server relays to it.
const ModuleRemoteRelay recipe.ModuleID = "model.remote-relay"

// RemoteInferenceDefinition returns the inference recipe of a remote
// model: the model manifest declares the provider document as its only
// component, the provider is the recipe's profile dependency, and one host
// relay node carries the prompt to the text under a request-scoped
// session, so the store compiles a component session plan for it.
func RemoteInferenceDefinition(modelID, providerID artifact.ID) (recipe.Definition, error) {
	relay := recipe.Node{ID: "relay", Module: ModuleRemoteRelay, Placement: recipe.PlacementHost, Session: recipe.SessionRequest}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskInference,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: modelID},
			{Role: recipe.DependencyProfile, Artifact: providerID},
		},
		[]recipe.Node{relay},
		nil,
		[]recipe.Input{{Name: "prompt", Data: recipe.DataText, Target: recipe.Endpoint{Node: relay.ID, Port: "prompt"}}},
		[]recipe.Output{{Name: "text", Data: recipe.DataText, Source: recipe.Endpoint{Node: relay.ID, Port: "text"}}},
	)
}

// RemoteModelDefinitionSchema identifies a hosted model's definition: the
// provider profile and the relay recipe stand where a local model's
// architecture and tensor inventory would, so evaluation plans bind it.
const RemoteModelDefinitionSchema = "overgo/remote-model-definition/v1"

// remoteModelDefinition: model + provider profile + relay recipe.
type remoteModelDefinition struct {
	Version  uint16      `json:"version"`
	Model    artifact.ID `json:"model"`
	Provider artifact.ID `json:"provider"`
	Recipe   artifact.ID `json:"recipe"`
	ID       artifact.ID `json:"-"`
}

var remoteModelDefinitionCodec = artifact.JSONDocumentCodec(
	"remote model definition", artifact.KindModelDefinition,
	"application/vnd.overgo.remote-model-definition+json", RemoteModelDefinitionSchema,
	func(value *remoteModelDefinition) error {
		if value.Version != artifact.InitialDocumentVersion || value.Model.Kind() != artifact.KindModel ||
			value.Provider.Kind() != artifact.KindProfile || value.Recipe.Kind() != artifact.KindRecipe {
			return errors.New("model recipe: invalid remote model definition")
		}
		return nil
	},
	func(value remoteModelDefinition) artifact.ID { return value.ID },
	func(value *remoteModelDefinition, id artifact.ID) { value.ID = id }, nil,
)

func remoteModelDefinitionLineage(value remoteModelDefinition) []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Model, value.Provider, value.Recipe)
}

// remoteRecipeProvider: the provider profile of a stored relay recipe;
// refused for any other recipe.
func remoteRecipeProvider(ctx context.Context, reader artifact.Reader, recipeID artifact.ID) (recipe.Definition, artifact.ID, error) {
	definition, err := recipe.RequireDefinition(ctx, reader, recipeID)
	if err != nil {
		return recipe.Definition{}, artifact.ID{}, err
	}
	provider, found := definition.PrimaryDependency(recipe.DependencyProfile)
	if !found || len(definition.Nodes) != 1 || definition.Nodes[0].Module != ModuleRemoteRelay {
		return recipe.Definition{}, artifact.ID{}, errors.New("model recipe: recipe is not a remote relay")
	}
	return definition, provider, nil
}

// PublishRemoteModelDefinition publishes the definition of the hosted
// model a relay recipe serves (idempotent) and returns its identity.
func PublishRemoteModelDefinition(ctx context.Context, repository artifact.Repository, recipeID artifact.ID) (artifact.ID, error) {
	if ctx == nil || repository == nil || recipeID.Kind() != artifact.KindRecipe {
		return artifact.ID{}, errors.New("model recipe: invalid remote definition publication")
	}
	definition, provider, err := remoteRecipeProvider(ctx, repository, recipeID)
	if err != nil {
		return artifact.ID{}, err
	}
	value, err := remoteModelDefinitionCodec.NewInitial(remoteModelDefinition{Model: definition.Model, Provider: provider, Recipe: definition.ID})
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := remoteModelDefinitionCodec.Batch("recipe/remote-definition/"+value.ID.String(), value, remoteModelDefinitionLineage(value), nil)
	if err != nil {
		return artifact.ID{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	if errors.Is(err, artifact.ErrNoChange) {
		err = nil
	}
	return value.ID, err
}

// requireRemoteModelDefinition: the binding check for a remote definition;
// model and recipe must be the stored relay's.
func requireRemoteModelDefinition(ctx context.Context, reader artifact.Reader, definitionID, modelID, runtimeRecipeID artifact.ID) error {
	value, err := remoteModelDefinitionCodec.RequireExactLineage(ctx, reader, definitionID, remoteModelDefinitionLineage)
	if err != nil || value.Model != modelID || value.Recipe != runtimeRecipeID {
		return errors.Join(err, errors.New("model recipe: remote definition differs from execution"))
	}
	definition, provider, err := remoteRecipeProvider(ctx, reader, value.Recipe)
	if err != nil || definition.Model != value.Model || provider != value.Provider {
		return errors.Join(err, errors.New("model recipe: remote definition differs from its relay recipe"))
	}
	return nil
}
