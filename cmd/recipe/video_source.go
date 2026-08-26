package main

import (
	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

func resolveDeclaredVideoSource(path string) (capabilitySource, error) {
	identity, err := hfrepo.InspectIdentity(path)
	if err != nil {
		return capabilitySource{}, err
	}
	source, err := modelrecipe.ResolveGenerationSource(recipe.TaskVideoGen, identity)
	if err != nil {
		return capabilitySource{}, err
	}
	inventory, err := imageGenInventory(path)
	if err != nil {
		return capabilitySource{}, err
	}
	return capabilitySource{inventory: inventory, define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
		definition, err := source.Definition(modelID, artifact.ID{})
		return definition, nil, err
	}}, nil
}
