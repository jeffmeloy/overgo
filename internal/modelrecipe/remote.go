package modelrecipe

import (
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
