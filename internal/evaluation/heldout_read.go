package evaluation

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

const (
	// objectiveReadUnit is what an objective's held-out budget counts: one trained candidate's read of the whole membership.
	objectiveReadUnit = "held-out read"
	// objectiveReadsIssued refuses nothing a run reaches: an objective outside a pool states no gain to size a budget by, so
	// its reads are counted, and a verdict is read beside how often its set has answered. A pool's confirmations stay bounded
	// (confirmOnPart).
	objectiveReadsIssued = math.MaxInt64
	// objectiveReader names who reads an objective's held-out membership, the consumer its charges are made for.
	objectiveReader = "overgo/objective-reader/"
)

// ChargeHeldoutRead charges one candidate's read of an objective's held-out membership and returns how many the membership
// has answered, this one among them. It runs before the records are handed over, so a read the store did not take is not
// made, and exposure a model was selected on is never left out of the record. A candidate charged before is not charged
// again: reading the same weights tells a proposer nothing more.
func ChargeHeldoutRead(ctx context.Context, store artifact.Repository, objective trainingprogram.ObjectiveDocument, heldout, candidate artifact.ID) (uint64, error) {
	if !heldout.Valid() || !candidate.Valid() {
		return 0, errors.New("evaluation: a held-out read names its membership and its candidate")
	}
	reader, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(objectiveReader+objective.ID.String()))
	if err != nil {
		return 0, err
	}
	grant, err := runrecord.NewBudget(objectiveReadUnit, heldout, objectiveReadsIssued, reader)
	if err != nil {
		return 0, err
	}
	key := poolKey + "read/" + grant.ID.String()
	batch, err := grant.Batch(key)
	if err != nil {
		return 0, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: reader})
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return 0, fmt.Errorf("evaluation: the read was not charged, so nothing was read: %w", err)
	}
	charge, err := runrecord.NewBudgetCharge(grant.ID, 1, reader, candidate.String())
	if err != nil {
		return 0, err
	}
	_, charges, err := runrecord.BudgetCharges(ctx, store, grant.ID)
	if err != nil {
		return 0, err
	}
	if slices.ContainsFunc(charges, func(committed runrecord.BudgetCharge) bool { return committed.ID == charge.ID }) {
		return uint64(len(charges)), nil
	}
	charged, err := charge.Batch(fmt.Sprintf("%s/%d", key, len(charges)+1))
	if err != nil {
		return 0, err
	}
	if _, err := artifact.Publish(ctx, store, charged); err != nil {
		return 0, fmt.Errorf("evaluation: the read was not charged, so nothing was read: %w", err)
	}
	return uint64(len(charges)) + 1, nil
}
