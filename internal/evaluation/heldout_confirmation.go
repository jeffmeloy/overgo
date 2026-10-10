package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
)

const (
	selectionReadingMedia  = "application/vnd.overgo.heldout-selection-reading+json"
	selectionReadingSchema = "overgo/heldout-selection-reading/v1"
	poolConfirmationMedia  = "application/vnd.overgo.heldout-confirmation+json"
	poolConfirmationSchema = "overgo/heldout-confirmation/v1"

	// The answers a confirmation has, which poolVerdicts counts.
	poolGain      = "gain"
	poolLoss      = "loss"
	poolUndecided = "not enough evidence"

	// poolKey prefixes the key of everything a pool publishes, before its
	// acceptance's identity.
	poolKey = "training/heldout-pool/"
	// poolConfirmer names the judge of an objective's confirmations, the
	// consumer their charges are made for. It is not the proposer, which the
	// promotion part is blinded from.
	poolConfirmer = "overgo/change-confirmer/"
	// poolChargePurpose is what a confirmation's charge is for, before the
	// candidate it names.
	poolChargePurpose = "confirmation of "
)

// CommittedPool is a held-out pool as the store holds it, for whoever fits
// and ranks candidates on it: the acceptance its confirmations are judged by,
// the partition that binds its parts, and the documents of its development
// and selection parts, each in its membership's order. The promotion part's
// documents are handed out by ConfirmOnPool alone.
type CommittedPool struct {
	Acceptance  PoolAcceptance
	Partition   runrecord.SplitPartition
	Development []string
	Selection   []string
}

// RequireHeldoutPool loads the pool an acceptance was fixed for.
func RequireHeldoutPool(ctx context.Context, reader artifact.Reader, acceptance artifact.ID) (CommittedPool, error) {
	var pool CommittedPool
	var err error
	if pool.Acceptance, err = poolAcceptanceCodec.Require(ctx, reader, acceptance); err != nil {
		return CommittedPool{}, err
	}
	content, found, err := artifact.ReadContent(ctx, reader, pool.Acceptance.Split)
	if err != nil || !found {
		return CommittedPool{}, errors.Join(fmt.Errorf("evaluation: the partition of pool %s is not committed", acceptance), err)
	}
	if pool.Partition, err = runrecord.ParseSplitPartition(content.Data); err != nil {
		return CommittedPool{}, err
	}
	if pool.Development, err = partDocuments(ctx, reader, pool.Partition.Development); err != nil {
		return CommittedPool{}, err
	}
	if pool.Selection, err = partDocuments(ctx, reader, pool.Partition.Selection); err != nil {
		return CommittedPool{}, err
	}
	return pool, nil
}

// partDocuments names the documents of one part of a pool, in its
// membership's order.
func partDocuments(ctx context.Context, reader artifact.Reader, shard artifact.ID) ([]string, error) {
	membership, found, err := dataset.LoadMembership(ctx, reader, shard)
	if err != nil || !found {
		return nil, errors.Join(fmt.Errorf("evaluation: the pool's part %s is not committed", shard), err)
	}
	documents := make([]string, len(membership.Records))
	for index, record := range membership.Records {
		documents[index] = record.ID
	}
	return documents, nil
}

// SelectionReading is one model read on the selection part of a pool by the
// pool's measure, before the base is taken from it: the mean log-likelihood of
// each document's tokens, a document a value in the order given. It is what a
// candidate is ranked by, and it names the candidate's weights by identity
// with the pool's acceptance, before any confirmation reads them.
type SelectionReading struct {
	ID         artifact.ID `json:"-"`
	Version    uint16      `json:"version"`
	Acceptance artifact.ID `json:"acceptance"`
	Model      artifact.ID `json:"model"`
	Documents  []string    `json:"documents"`
	Values     []float64   `json:"values"`
}

var selectionReadingCodec = artifact.JSONDocumentCodec(
	"held-out selection reading", artifact.KindEvaluation, selectionReadingMedia, selectionReadingSchema,
	func(value *SelectionReading) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Acceptance.Kind() != artifact.KindEvidence ||
			value.Model.Kind() != artifact.KindModel || len(value.Documents) == 0 || len(value.Documents) != len(value.Values) {
			return errors.New("evaluation: invalid held-out selection reading")
		}
		return nil
	},
	func(value SelectionReading) artifact.ID { return value.ID },
	func(value *SelectionReading, id artifact.ID) { value.ID = id },
	func(value SelectionReading) SelectionReading {
		value.Documents, value.Values = slices.Clone(value.Documents), slices.Clone(value.Values)
		return value
	},
)

// NewSelectionReading records what a model scored on the selection part of
// the pool an acceptance was fixed for, and stages it under key.
func NewSelectionReading(key string, acceptance, model artifact.ID, scores []RecordScore) (SelectionReading, artifact.Batch, error) {
	reading := SelectionReading{Acceptance: acceptance, Model: model}
	for _, score := range scores {
		reading.Documents, reading.Values = append(reading.Documents, score.Record), append(reading.Values, score.Value)
	}
	reading, err := selectionReadingCodec.NewInitial(reading)
	if err != nil {
		return SelectionReading{}, artifact.Batch{}, err
	}
	batch, err := selectionReadingCodec.Batch(key, reading, artifact.DependencyLineage(reading.ID, acceptance), nil)
	return reading, batch, err
}

// RequireSelectionReading loads one committed selection reading.
func RequireSelectionReading(ctx context.Context, reader artifact.Reader, id artifact.ID) (SelectionReading, error) {
	return selectionReadingCodec.Require(ctx, reader, id)
}

// PoolConfirmation is one confirmation as a pool records it: the acceptance
// it was judged by, the candidate, the charge it cost and the verdict, a gain,
// a loss or not enough evidence. The statistic the verdict was read from is
// recorded nowhere: the acceptance divides its false-gain rate over what a
// proposer told verdicts could reach, and one that could read the statistic
// would be told more.
type PoolConfirmation struct {
	ID         artifact.ID `json:"-"`
	Version    uint16      `json:"version"`
	Acceptance artifact.ID `json:"acceptance"`
	Candidate  artifact.ID `json:"candidate"`
	Charge     artifact.ID `json:"charge"`
	Verdict    string      `json:"verdict"`
}

var poolConfirmationCodec = artifact.JSONDocumentCodec(
	"held-out confirmation", artifact.KindEvaluation, poolConfirmationMedia, poolConfirmationSchema,
	func(value *PoolConfirmation) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Acceptance.Kind() != artifact.KindEvidence ||
			!value.Candidate.Valid() || value.Charge.Kind() != artifact.KindEvidence ||
			!slices.Contains([]string{poolGain, poolLoss, poolUndecided}, value.Verdict) {
			return errors.New("evaluation: invalid held-out confirmation")
		}
		return nil
	},
	func(value PoolConfirmation) artifact.ID { return value.ID },
	func(value *PoolConfirmation, id artifact.ID) { value.ID = id }, nil,
)

// ConfirmOnPool has the promotion part of a pool answer for one candidate. The
// candidate and the acceptance are named by identity before anything is read,
// and without both the confirmation is refused. A confirmation is charged to
// the promotion part's budget before its documents are handed to read, which
// returns the pool's measure for each of them, the candidate less the base; the
// verdict is read from those by the acceptance's rule and published, and it
// returns with what the budget has left and nothing else of the reading.
//
// What is left is counted from the charges the store holds, so it is the same
// in any process. A candidate charged for before is not charged again: its
// recorded verdict is returned, or read now when the reading that owed it did
// not finish. When nothing is left a confirmation is refused, and the pool
// answers again only as a fresh partition.
func ConfirmOnPool(ctx context.Context, store artifact.Repository, acceptanceID, candidate artifact.ID, read func(documents []string) ([]float64, error)) (PoolConfirmation, uint64, error) {
	pool, err := RequireHeldoutPool(ctx, store, acceptanceID)
	if err != nil || !candidate.Valid() {
		return PoolConfirmation{}, 0, errors.Join(errors.New("evaluation: a confirmation names its candidate and the acceptance it is judged by before it reads anything"), err)
	}
	acceptance := pool.Acceptance
	promotion, err := partDocuments(ctx, store, pool.Partition.Promotion)
	if err != nil || uint64(len(promotion)) != acceptance.Documents {
		return PoolConfirmation{}, 0, errors.Join(fmt.Errorf("evaluation: the promotion part of pool %s holds %d documents and its acceptance was fixed for %d", acceptance.ID, len(promotion), acceptance.Documents), err)
	}
	// The grant is the one the pool published: named by what it grants, and loaded with its charges.
	granted, err := runrecord.NewBudget(poolUnit, pool.Partition.Promotion, acceptance.Confirmations, acceptance.ID)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	budget, charges, err := runrecord.BudgetCharges(ctx, store, granted.ID)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	confirmer, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(poolConfirmer+acceptance.Objective.String()))
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	charge, err := runrecord.NewBudgetCharge(budget.ID, 1, confirmer, poolChargePurpose+candidate.String())
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	remaining, err := runrecord.BudgetBalance(budget, charges)
	if err == nil {
		err = runrecord.ValidateBlinding(pool.Partition, budget, charges)
	}
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	if slices.ContainsFunc(charges, func(committed runrecord.BudgetCharge) bool { return committed.ID == charge.ID }) {
		verdicts, err := store.Children(ctx, charge.ID)
		if err != nil {
			return PoolConfirmation{}, 0, err
		}
		for _, edge := range verdicts {
			if recorded, err := poolConfirmationCodec.Require(ctx, store, edge.Child); err == nil {
				return recorded, remaining, nil
			}
		}
	} else {
		if remaining == 0 {
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: the promotion part of pool %s has answered the %d confirmations its %d documents allow; a further one needs a fresh partition",
				acceptance.ID, acceptance.Confirmations, acceptance.Documents)
		}
		// The key counts the charge. The store holds one content under a key,
		// so of two confirmations that would each be the next, one is charged
		// and the other is refused by the store.
		batch, err := charge.Batch(fmt.Sprintf("%s%s/confirmation/%d", poolKey, acceptance.ID, len(charges)+1))
		if err != nil {
			return PoolConfirmation{}, 0, err
		}
		if _, held, err := store.Artifact(ctx, confirmer); err != nil {
			return PoolConfirmation{}, 0, err
		} else if !held {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: confirmer})
		}
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: the confirmation was not charged, so nothing was read: %w", err)
		}
		remaining--
	}
	differences, err := read(promotion)
	if err != nil || len(differences) != len(promotion) {
		return PoolConfirmation{}, 0, errors.Join(fmt.Errorf("evaluation: the candidate was read on %d of the promotion part's %d documents; its confirmation is charged and its verdict is owed", len(differences), len(promotion)), err)
	}
	confirmation, err := poolConfirmationCodec.NewInitial(PoolConfirmation{
		Acceptance: acceptance.ID, Candidate: candidate, Charge: charge.ID, Verdict: poolVerdict(differences, acceptance.Critical),
	})
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	batch, err := poolConfirmationCodec.Batch(poolKey+acceptance.ID.String()+"/verdict/"+charge.ID.String(), confirmation,
		artifact.DependencyLineage(confirmation.ID, acceptance.ID, charge.ID), nil)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return PoolConfirmation{}, 0, err
	}
	return confirmation, remaining, nil
}

// poolVerdict reads a confirmation from the measure's paired differences, one
// a document of the promotion part: a gain when their mean over its standard
// error passes the acceptance's statistic, a loss when it passes it downward,
// and otherwise not enough evidence. Differences that do not vary have no
// error to divide by: the statistic is then without bound on the mean's side,
// and undefined, passing neither way, when the mean is zero.
func poolVerdict(differences []float64, critical float64) string {
	scores := make([]RecordScore, len(differences))
	var mean float64
	for index, difference := range differences {
		scores[index].Value = difference
		mean += difference / float64(len(differences))
	}
	switch statistic := mean / meanError(scores); {
	case statistic > critical:
		return poolGain
	case statistic < -critical:
		return poolLoss
	}
	return poolUndecided
}
