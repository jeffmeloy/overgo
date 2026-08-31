package modelrecipe

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// RollbackActivation returns a model task to the recipe its retired
// activation superseded. Rollback is not a state reversal: the predecessor
// re-earns active through a fresh verified gate/run bound to its own
// identity, the rollback activation cites the retirement it answers, and
// the retired recipe stays superseded on the record.
func RollbackActivation(
	ctx context.Context,
	store artifact.Repository,
	model artifact.ID,
	task recipe.Task,
	verification Verification,
	tier recipe.EvidenceTier,
	reason string,
) error {
	if strings.TrimSpace(reason) == "" || strings.TrimSpace(reason) != reason || !tier.Valid() {
		return errors.New("model recipe: rollback decision is invalid")
	}
	retiredID, active, err := artifact.ResolveAlias(ctx, store, activeAlias(model, task))
	if err != nil {
		return err
	}
	if !active {
		return errors.New("model recipe: rollback requires a retired activation")
	}
	retiredEvent, err := currentEvent(ctx, store, retiredID)
	if err != nil {
		return err
	}
	if retiredEvent.To != recipe.StatusSuperseded {
		return fmt.Errorf(
			"model recipe: the active recipe is %q; retire it with failed proof before rolling back",
			retiredEvent.To,
		)
	}
	activation := retiredEvent
	for activation.To != recipe.StatusActive {
		if activation.PreviousEvent == nil {
			return errors.New("model recipe: the retired recipe was never active; nothing to roll back")
		}
		previous, found, readErr := recipe.ReadLifecycleEvent(ctx, store, *activation.PreviousEvent)
		if readErr != nil {
			return readErr
		}
		if !found {
			return errors.New("model recipe: retired lifecycle history is absent or incompatible")
		}
		activation = previous
	}
	if activation.Supersedes == nil {
		return errors.New("model recipe: the retired activation superseded nothing; there is no predecessor to restore")
	}
	predecessor, err := recipe.RequireDefinition(ctx, store, *activation.Supersedes)
	if err != nil {
		return err
	}
	if predecessor.Model != model || predecessor.Task != task {
		return errors.New("model recipe: rollback predecessor subject mismatch")
	}
	_, _, err = ActivateVerified(
		ctx, store, "recipe/rollback/"+predecessor.ID.String(), predecessor,
		verification, tier, reason, []artifact.ID{retiredEvent.ID}, &retiredID,
	)
	return err
}
