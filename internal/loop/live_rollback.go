package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

const (
	// LiveRollbackMediaType identifies live rollback causal records.
	LiveRollbackMediaType = "application/vnd.overgo.live-rollback+json"
	// LiveRollbackSchema identifies the rollback causal contract.
	LiveRollbackSchema = "overgo/live-rollback/v1"
)

// LiveRollbackRecord is the published causal link of one live rollback:
// which recipe was retired, which predecessor returned to service, and the
// exact breaker transition whose regression evidence caused it.
type LiveRollbackRecord struct {
	Model      artifact.ID `json:"model"`
	Task       recipe.Task `json:"task"`
	Retired    artifact.ID `json:"retired"`
	Restored   artifact.ID `json:"restored"`
	Transition artifact.ID `json:"transition"`
	Reason     string      `json:"reason"`
}

// RollbackQuarantinedCandidate rolls the active alias back after a
// containment-level breaker transition. The decision binds the exact
// recipe the breaker judged: when the active alias has already moved — a
// concurrent promotion, rollback, or operator action won — the stale
// decision refuses and preserves its evidence instead of overwriting the
// newer head. Retirement, predecessor reactivation, and every alias
// compare-and-set run through the modelrecipe promotion owner, and the
// causal record publishes only after the rollback holds, citing the
// transition, the retired recipe, and the restored predecessor.
func RollbackQuarantinedCandidate(
	ctx context.Context,
	store artifact.Repository,
	model artifact.ID,
	task recipe.Task,
	judged artifact.ID,
	transitionEvidence artifact.ID,
	retirement modelrecipe.Verification,
	reactivation modelrecipe.Verification,
	reason string,
) (artifact.ID, error) {
	content, found, err := artifact.ReadContent(ctx, store, transitionEvidence)
	if err != nil {
		return artifact.ID{}, err
	}
	if !found || content.Descriptor.MediaType != CircuitBreakerMediaType {
		return artifact.ID{}, errors.New("loop: rollback requires the published breaker transition evidence")
	}
	var transition CircuitBreakerTransition
	if err := json.Unmarshal(content.Data, &transition); err != nil {
		return artifact.ID{}, err
	}
	if transition.Level != BreakerQuarantine && transition.Level != BreakerOperatorStop {
		return artifact.ID{}, fmt.Errorf(
			"loop: breaker level %q is not a containment transition; rollback refuses", transition.Level,
		)
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, model, task)
	if err != nil {
		return artifact.ID{}, err
	}
	if !active || activation.Definition.ID != judged {
		return artifact.ID{}, errors.New(
			"loop: the active recipe moved past the judged candidate; the stale rollback decision preserves its evidence and changes nothing",
		)
	}
	if err := modelrecipe.RetireActiveCapability(
		ctx, store, activation.Definition, retirement, reason,
	); err != nil {
		return artifact.ID{}, err
	}
	if err := modelrecipe.RollbackActivation(
		ctx, store, model, task, reactivation, recipe.EvidenceVerified, reason,
	); err != nil {
		return artifact.ID{}, err
	}
	restored, stillActive, err := modelrecipe.ActiveRecord(ctx, store, model, task)
	if err != nil || !stillActive {
		return artifact.ID{}, errors.Join(errors.New("loop: rollback did not restore an active predecessor"), err)
	}
	record := LiveRollbackRecord{
		Model: model, Task: task, Retired: judged,
		Restored: restored.Definition.ID, Transition: transitionEvidence, Reason: reason,
	}
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: LiveRollbackMediaType, Schema: LiveRollbackSchema,
	}
	recordContent, err := artifact.JSONContent(contract, record)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"live-safety/rollback/"+recordContent.Descriptor.ID.String(),
		[]artifact.Content{recordContent},
		artifact.UniqueDependencyLineage(
			recordContent.Descriptor.ID, transitionEvidence, judged, restored.Definition.ID, model,
		),
		nil,
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return recordContent.Descriptor.ID, nil
}
