package modelrecipe

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
)

// MaterializePrototypeTrial assembles the one atomic batch a successful
// prototype trial publishes: the trained model's inventory and immutable
// tensor facts, the resolved model definition binding those exact bytes to
// the prototype's declared architecture and profile, and the candidate
// recipes citing the resolved definition, the trial's run receipt, and the
// admission evidence. Nothing here moves an alias: a prototype never becomes
// a model by alias mutation alone — activation stays with the recipe
// lifecycle, and a batch that would carry an alias refuses.
func MaterializePrototypeTrial(
	trial PrototypeTrial,
	resolved ResolvedModelDefinition,
	inventory modelartifact.Inventory,
	trialRun artifact.ID,
	candidateRecipes []artifact.ID,
) (artifact.Batch, error) {
	if resolved.Document.Architecture != trial.Prototype.Architecture {
		return artifact.Batch{}, errors.New(
			"model recipe: materialized bytes must bind the prototype's exact architecture",
		)
	}
	if resolved.Profile.ID != trial.Prototype.ArchitectureProfile {
		return artifact.Batch{}, errors.New(
			"model recipe: materialized bytes must bind the prototype's exact architecture profile",
		)
	}
	if trialRun.Kind() != artifact.KindRun {
		return artifact.Batch{}, errors.New("model recipe: materialization requires the trial's exact run receipt")
	}
	if len(candidateRecipes) == 0 {
		return artifact.Batch{}, errors.New("model recipe: materialization requires at least one candidate recipe")
	}
	batch, err := resolved.Batch("model-prototype/trial/"+trial.Prototype.ID.String(), inventory)
	if err != nil {
		return artifact.Batch{}, err
	}
	if len(batch.Aliases) != 0 {
		return artifact.Batch{}, errors.New("model recipe: a prototype never becomes a model by alias mutation")
	}
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: trialRun}, artifact.Descriptor{ID: trial.Admission},
	)
	for _, candidate := range candidateRecipes {
		if candidate.Kind() != artifact.KindRecipe {
			return artifact.Batch{}, errors.New("model recipe: candidate recipes must be exact recipe identities")
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: candidate})
		batch.Lineage = append(batch.Lineage,
			artifact.DependencyLineage(candidate, resolved.Document.ID, trialRun, trial.Admission)...)
	}
	if err := batch.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
}
