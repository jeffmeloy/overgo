package modelrecipe

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const activityModelSlot uint32 = 1 // Slot zero remains the transcription model.

// SegmentedTranscriptionDefinition composes the canonical offline activity and
// transcription components. Activity owns boundaries and finalization; the
// transcription stage consumes those nonoverlapping sample spans unchanged.
func SegmentedTranscriptionDefinition(base, activity recipe.Definition) (recipe.Definition, error) {
	if err := base.ValidateIdentity(); err != nil {
		return recipe.Definition{}, err
	}
	if err := activity.ValidateIdentity(); err != nil {
		return recipe.Definition{}, err
	}
	canonical, err := canonicalTranscription(base.Dependencies)
	if err != nil || canonical.ID != base.ID {
		return recipe.Definition{}, errors.Join(errors.New("segmented transcription: base topology differs"), err)
	}
	model, _ := activity.PrimaryDependency(recipe.DependencyModel)
	profile, _ := activity.PrimaryDependency(recipe.DependencyProcessorProfile)
	inventory, _ := activity.PrimaryDependency(recipe.DependencyTensorInventory)
	expected, err := ActivityDefinition(model, profile, inventory)
	if err != nil || expected.ID != activity.ID {
		return recipe.Definition{}, errors.Join(errors.New("segmented transcription: activity topology differs"), err)
	}
	dependencies := slices.Clone(base.Dependencies)
	for _, dependency := range activity.Dependencies {
		dependency.Slot = activityModelSlot
		dependencies = append(dependencies, dependency)
	}
	dependencies = append(dependencies,
		recipe.Dependency{Role: recipe.DependencyExecutionRecipe, Artifact: base.ID},
		recipe.Dependency{Role: recipe.DependencyExecutionRecipe, Slot: activityModelSlot, Artifact: activity.ID})
	activityNode, transcriptionNode := activity.Nodes[0], base.Nodes[0]
	activityNode.ModelSlot = activityModelSlot
	transcriptionNode.Module = ModuleTranscribeSegments
	return recipe.NewDefinitionWithDependencies(recipe.TaskTranscription, dependencies,
		[]recipe.Node{activityNode, transcriptionNode},
		[]recipe.Edge{{From: recipe.Endpoint{Node: activityNode.ID, Port: "segments"}, To: recipe.Endpoint{Node: transcriptionNode.ID, Port: "segments"}}},
		activity.Inputs, base.Outputs)
}

// TranscriptionComponents validates the complete topology and returns exact
// standalone component definitions. An empty activity definition denotes one
// whole-clip transcription stage; unrecognized or extra dependencies refuse.
func TranscriptionComponents(definition recipe.Definition) (base, activity recipe.Definition, err error) {
	if err = definition.ValidateIdentity(); err != nil {
		return base, activity, err
	}
	var baseDependencies, activityDependencies []recipe.Dependency
	for _, dependency := range definition.Dependencies {
		switch dependency.Slot {
		case 0:
			baseDependencies = append(baseDependencies, dependency)
		case activityModelSlot:
			dependency.Slot = 0
			activityDependencies = append(activityDependencies, dependency)
		default:
			return base, activity, errors.New("transcription: unsupported component slot")
		}
	}
	base, err = canonicalTranscription(baseDependencies)
	if err != nil {
		return base, activity, err
	}
	expected := base
	if len(activityDependencies) != 0 {
		candidate := recipe.Definition{Dependencies: activityDependencies}
		model, _ := candidate.PrimaryDependency(recipe.DependencyModel)
		profile, _ := candidate.PrimaryDependency(recipe.DependencyProcessorProfile)
		inventory, _ := candidate.PrimaryDependency(recipe.DependencyTensorInventory)
		activity, err = ActivityDefinition(model, profile, inventory)
		if err == nil {
			expected, err = SegmentedTranscriptionDefinition(base, activity)
		}
	}
	if err != nil || expected.ID != definition.ID {
		return recipe.Definition{}, recipe.Definition{}, errors.Join(errors.New("transcription: complete topology differs"), err)
	}
	return base, activity, nil
}

func canonicalTranscription(dependencies []recipe.Dependency) (recipe.Definition, error) {
	d := recipe.Definition{Dependencies: dependencies}
	roles := [...]recipe.DependencyRole{recipe.DependencyModel, recipe.DependencyProfile, recipe.DependencyProcessorProfile, recipe.DependencyTokenizer, recipe.DependencyTensorInventory}
	var ids [len(roles)]artifact.ID
	for index, role := range roles {
		ids[index], _ = d.PrimaryDependency(role)
	}
	base, err := TranscriptionDefinition(ids[0], ids[1], ids[2], ids[3], ids[4])
	if err != nil {
		return recipe.Definition{}, err
	}
	if checkpoint, ok := d.PrimaryDependency(recipe.DependencyCheckpoint); ok {
		return AdaptedTranscriptionDefinition(base, checkpoint)
	}
	return base, nil
}
