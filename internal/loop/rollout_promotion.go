package loop

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// RolloutPromotionClosure names the complete evidence a promotion must
// present: the rollout plan, the projection reading taken at the rollout's
// baseline head, and the reading taken at the boundary head. Coverage and
// the rollback contract are read from those bindings — nothing is asserted
// out of band.
type RolloutPromotionClosure struct {
	Plan             artifact.ID
	BaselineReading  artifact.ID
	CandidateReading artifact.ID
}

func loadRolloutReading(
	ctx context.Context, store artifact.Reader, id artifact.ID,
) (evaluation.RolloutProjection, error) {
	content, found, err := artifact.ReadContent(ctx, store, id)
	if err != nil {
		return evaluation.RolloutProjection{}, err
	}
	if !found {
		return evaluation.RolloutProjection{}, fmt.Errorf("loop: rollout reading %s is not committed", id)
	}
	return evaluation.ParseRolloutProjection(content.Data)
}

// PromoteRolloutCandidate activates a rollout's candidate only after the
// required evidence closure is present: both projection readings bind the
// exact plan, the boundary reading strictly follows the baseline reading,
// the new observations between them meet the plan's observation contract,
// the rollback contract is committed, and the live active recipe is the
// plan's own baseline. Offline evaluation admitted the candidate to its
// cohort; universal activation happens only here, and the compare-and-set
// alias transition itself runs through the modelrecipe promotion owner.
func PromoteRolloutCandidate(
	ctx context.Context,
	store artifact.Repository,
	closure RolloutPromotionClosure,
	verification modelrecipe.Verification,
	reason string,
) error {
	content, found, err := artifact.ReadContent(ctx, store, closure.Plan)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("loop: rollout plan %s is not committed", closure.Plan)
	}
	plan, err := runrecord.ParseRolloutPlan(content.Data)
	if err != nil {
		return err
	}
	baseline, err := loadRolloutReading(ctx, store, closure.BaselineReading)
	if err != nil {
		return err
	}
	boundary, err := loadRolloutReading(ctx, store, closure.CandidateReading)
	if err != nil {
		return err
	}
	if baseline.Plan != plan.ID || boundary.Plan != plan.ID {
		return errors.New("loop: promotion readings bind another rollout plan")
	}
	if boundary.Sequence <= baseline.Sequence {
		return errors.New("loop: the boundary reading does not follow the baseline reading")
	}
	if boundary.Coverage < baseline.Coverage {
		return errors.New("loop: the boundary reading lost recorded observations")
	}
	observed := boundary.Coverage - baseline.Coverage
	if observed < plan.Observation.MinObservations {
		return fmt.Errorf(
			"loop: %d observations between readings have not met the observation contract of %d",
			observed, plan.Observation.MinObservations,
		)
	}
	if _, present, err := store.Artifact(ctx, plan.Rollback); err != nil || !present {
		return errors.Join(err, errors.New("loop: the rollout rollback contract is absent"))
	}
	definition, err := recipe.RequireDefinition(ctx, store, plan.Candidate)
	if err != nil {
		return err
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, definition.Model, definition.Task)
	if err != nil {
		return err
	}
	if !active || activation.Definition.ID != plan.Baseline {
		return errors.New("loop: the live active recipe is not the rollout baseline; promotion would supersede something the rollout never compared")
	}
	return modelrecipe.ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceVerified, reason,
	)
}
