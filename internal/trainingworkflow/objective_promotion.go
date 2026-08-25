package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

// PromoteObjectiveAdaptive republishes a registered objective one rung
// up the validation ladder, from declared to adaptive-evidence, gated
// on committed proof rather than caller assertion: every named
// observation must be a SUCCEEDED training session observation whose
// recipe's objective dependency is this exact objective. The promoted
// document carries the observations in its evidence set, and the
// registered alias rebinds with compare-and-set supersession.
func PromoteObjectiveAdaptive(
	ctx context.Context,
	repository artifact.Repository,
	aliasName string,
	observations []artifact.ID,
) (trainingprogram.ObjectiveDocument, error) {
	if ctx == nil || repository == nil {
		return trainingprogram.ObjectiveDocument{}, errors.New("training workflow: nil promotion context or repository")
	}
	if len(observations) == 0 {
		return trainingprogram.ObjectiveDocument{}, errors.New("training workflow: promotion requires at least one training observation")
	}
	currentID, bound, err := repository.ResolveAlias(ctx, aliasName)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	if !bound {
		return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: objective alias %q is absent", aliasName)
	}
	current, err := trainingprogram.LoadObjective(ctx, repository, currentID)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	if current.Authority != trainingprogram.ObjectiveDeclared {
		return trainingprogram.ObjectiveDocument{}, fmt.Errorf(
			"training workflow: objective %q holds %q; only a declared objective climbs to adaptive-evidence", aliasName, current.Authority)
	}
	for _, observationID := range observations {
		observation, err := runrecord.RequireServingObservation(ctx, repository, observationID)
		if err != nil {
			return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: promotion evidence %s: %w", observationID, err)
		}
		if observation.Task != recipe.TaskTraining || observation.Outcome != runrecord.OutcomeSucceeded {
			return trainingprogram.ObjectiveDocument{}, fmt.Errorf(
				"training workflow: promotion evidence %s is task %q outcome %q, need a succeeded training session", observationID, observation.Task, observation.Outcome)
		}
		definition, err := loadRecipeDefinition(ctx, repository, observation.Recipe)
		if err != nil {
			return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: promotion evidence %s recipe: %w", observationID, err)
		}
		grounded, ok := definition.PrimaryDependency(recipe.DependencyObjective)
		if !ok || grounded != current.ID {
			return trainingprogram.ObjectiveDocument{}, fmt.Errorf(
				"training workflow: promotion evidence %s trained under a different objective", observationID)
		}
	}
	spec := current.ObjectiveSpec
	spec.Authority = trainingprogram.ObjectiveAdaptive
	spec.Evidence = slices.Clone(spec.Evidence)
	for _, observationID := range observations {
		if !slices.Contains(spec.Evidence, observationID) {
			spec.Evidence = append(spec.Evidence, observationID)
		}
	}
	promoted, err := trainingprogram.NewObjective(spec)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	content, err := promoted.Content()
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	batch := artifact.Batch{
		Key:      "training-objective-promotion/" + promoted.ID.String(),
		Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{{
			Name: aliasName, Target: promoted.ID, Previous: artifact.IDPointer(current.ID),
		}},
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return trainingprogram.ObjectiveDocument{}, err
	}
	return promoted, nil
}

// PromoteObjectiveApproved republishes an adaptive-evidence objective
// at approved, the top of the validation ladder, gated on committed
// PASSED evaluation reports: every named report must resolve to a
// typed evaluation report artifact in the store whose verdict is
// passed. An arbitrary evidence identity, a failed report, or an
// objective below adaptive is refused -- approved is earned by
// executable evaluation, never asserted.
func PromoteObjectiveApproved(
	ctx context.Context,
	repository artifact.Repository,
	aliasName string,
	reports []artifact.ID,
) (trainingprogram.ObjectiveDocument, error) {
	if ctx == nil || repository == nil {
		return trainingprogram.ObjectiveDocument{}, errors.New("training workflow: nil promotion context or repository")
	}
	if len(reports) == 0 {
		return trainingprogram.ObjectiveDocument{}, errors.New("training workflow: approval requires at least one passed evaluation report")
	}
	currentID, bound, err := repository.ResolveAlias(ctx, aliasName)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	if !bound {
		return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: objective alias %q is absent", aliasName)
	}
	current, err := trainingprogram.LoadObjective(ctx, repository, currentID)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	if current.Authority != trainingprogram.ObjectiveAdaptive {
		return trainingprogram.ObjectiveDocument{}, fmt.Errorf(
			"training workflow: objective %q holds %q; only adaptive-evidence climbs to approved", aliasName, current.Authority)
	}
	for _, reportID := range reports {
		passed, err := evaluation.CommittedReportPassed(ctx, repository, reportID)
		if err != nil {
			return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: approval evidence %s: %w", reportID, err)
		}
		if !passed {
			return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: approval evidence %s did not pass", reportID)
		}
	}
	spec := current.ObjectiveSpec
	spec.Authority = trainingprogram.ObjectiveApproved
	spec.Evidence = slices.Clone(spec.Evidence)
	for _, reportID := range reports {
		if !slices.Contains(spec.Evidence, reportID) {
			spec.Evidence = append(spec.Evidence, reportID)
		}
	}
	promoted, err := trainingprogram.NewObjective(spec)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	content, err := promoted.Content()
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	batch := artifact.Batch{
		Key:      "training-objective-approval/" + promoted.ID.String(),
		Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{{
			Name: aliasName, Target: promoted.ID, Previous: artifact.IDPointer(current.ID),
		}},
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return trainingprogram.ObjectiveDocument{}, err
	}
	return promoted, nil
}

// loadRecipeDefinition reads one recipe definition by exact identity.
func loadRecipeDefinition(ctx context.Context, reader artifact.Reader, id artifact.ID) (recipe.Definition, error) {
	content, ok, err := artifact.ReadContent(ctx, reader, id)
	if err != nil {
		return recipe.Definition{}, err
	}
	if !ok || content.Descriptor.Schema != recipe.Schema {
		return recipe.Definition{}, errors.New("training workflow: recipe definition content is absent or incompatible")
	}
	definition, err := recipe.ParseDefinition(content.Data)
	if err != nil {
		return recipe.Definition{}, err
	}
	if recipe.DefinitionDocumentContract().ValidateContent(content, id) != nil || definition.ID != id {
		return recipe.Definition{}, errors.New("training workflow: recipe definition identity differs")
	}
	return definition, nil
}
