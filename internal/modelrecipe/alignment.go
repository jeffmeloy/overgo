package modelrecipe

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// AlignmentDefinition binds a standalone, unadapted recognizer and an explicit
// alignment declaration to the existing typed audio/transcript contract.
func AlignmentDefinition(base recipe.Definition, alignmentProfile artifact.ID) (recipe.Definition, error) {
	if err := base.ValidateIdentity(); err != nil {
		return recipe.Definition{}, err
	}
	canonical, err := canonicalTranscription(base.Dependencies)
	_, adapted := base.PrimaryDependency(recipe.DependencyCheckpoint)
	if err != nil || canonical.ID != base.ID || adapted || alignmentProfile.Kind() != artifact.KindProfile {
		return recipe.Definition{}, errors.New("alignment: standalone base recognizer and alignment profile required")
	}
	dependencies := append(slices.Clone(base.Dependencies),
		recipe.Dependency{Role: recipe.DependencyExecutionRecipe, Artifact: base.ID},
		recipe.Dependency{Role: recipe.DependencyDerivationProfile, Artifact: alignmentProfile})
	node := recipe.Node{ID: "align", Module: ModuleAlignAudio, Placement: recipe.PlacementHost,
		Session: recipe.SessionCapacity, Residency: recipe.ResidencyHostCache}
	return recipe.NewDefinitionWithDependencies(recipe.TaskAlignment, dependencies, []recipe.Node{node}, nil,
		[]recipe.Input{
			{Name: "audio", Data: recipe.DataAudio, Target: recipe.Endpoint{Node: node.ID, Port: "audio"}},
			{Name: "transcription", Data: recipe.DataTranscription, Target: recipe.Endpoint{Node: node.ID, Port: "transcription"}},
		}, []recipe.Output{{Name: "alignment", Data: recipe.DataTimestampedAlignment, Source: recipe.Endpoint{Node: node.ID, Port: "alignment"}}})
}
