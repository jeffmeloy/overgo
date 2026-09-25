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
		definition, err := recipe.RequireDefinition(ctx, repository, observation.Recipe)
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
		Aliases:  []artifact.AliasBinding{artifact.AliasMove(aliasName, promoted.ID, current.ID)},
	}
	if _, err := artifact.Publish(ctx, repository, batch); err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	return promoted, nil
}

// PromoteObjectiveApproved republishes an adaptive-evidence objective
// at approved, the top of the validation ladder, gated on committed
// PASSED evaluation reports: every named report must resolve to a
// typed evaluation report artifact in the store whose verdict is
// passed, and at least one must be a held-out verdict for this
// objective: the trained model beat its base on records its training
// never saw. An arbitrary evidence identity, a failed report, a
// benchmark report alone, or an objective below adaptive is refused --
// approved is earned by executable held-out evaluation, never asserted.
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
	heldout := false
	for _, reportID := range reports {
		passed, isVerdict, err := approvalReportPassed(ctx, repository, reportID, current)
		if err != nil {
			return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: approval evidence %s: %w", reportID, err)
		}
		if !passed {
			return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: approval evidence %s did not pass", reportID)
		}
		heldout = heldout || isVerdict
	}
	if !heldout {
		return trainingprogram.ObjectiveDocument{}, fmt.Errorf(
			"training workflow: approving %q requires a passed held-out verdict scored on its held-out split", aliasName)
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
		Aliases:  []artifact.AliasBinding{artifact.AliasMove(aliasName, promoted.ID, current.ID)},
	}
	if _, err := artifact.Publish(ctx, repository, batch); err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	return promoted, nil
}

// approvalReportPassed reads one approval report: a held-out verdict is
// re-derived for current, any other report must be a typed evaluation
// report. It says which kind it read.
func approvalReportPassed(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	current trainingprogram.ObjectiveDocument,
) (passed, heldout bool, err error) {
	descriptor, found, err := reader.Artifact(ctx, id)
	if err != nil {
		return false, false, err
	}
	if found && evaluation.IsHeldoutVerdict(descriptor) {
		passed, err := evaluation.CommittedHeldoutVerdictPassed(ctx, reader, id, current)
		return passed, true, err
	}
	passed, err = evaluation.CommittedReportPassed(ctx, reader, id)
	return passed, false, err
}
