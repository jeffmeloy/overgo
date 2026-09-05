package mediacapability

import (
	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

func resolveDeclaredVideoSource(path string) (Source, error) {
	identity, err := hfrepo.InspectIdentity(path)
	if err != nil {
		return Source{}, err
	}
	source, err := modelrecipe.ResolveGenerationSource(recipe.TaskVideoGen, identity)
	if err != nil {
		return Source{}, err
	}
	inventory, err := imageGenInventory(path)
	if err != nil {
		return Source{}, err
	}
	return Source{Inventory: inventory, Define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
		definition, err := modelrecipe.GenerationDefinition(source.Prepare, modelID, artifact.ID{})
		return definition, nil, err
	}}, nil
}
