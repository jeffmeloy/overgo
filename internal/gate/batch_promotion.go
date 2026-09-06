package gate

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

// checkpointPromotion: one run's flush decision over the declared key;
// pending obligations are committed by the final record whatever the
// outcome, the promotion only with a successful full gate.
type checkpointPromotion struct {
	key      string
	pending  []runrecord.CheckpointObligation
	promoted []artifact.ID
	flushed  bool
	reason   plan.FlushReason
}

// decideCheckpointPromotion: outstanding obligations of the key that still
// match this candidate's memo inputs seed the accumulator, every accepted
// checkpoint of this run without a matching obligation becomes a pending one,
// and the declared bounds decide the flush; a stale obligation is neither
// seeded nor promoted and is audited.
func (g *gateContext) decideCheckpointPromotion(ctx context.Context, now time.Time) error {
	batch := g.verificationBatch
	if batch == nil || batch.Flush == nil {
		return nil
	}
	if g.storePath == "" {
		return errors.New("acceptance: a flush declaration requires the gate store")
	}
	store, err := overgodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return err
	}
	defer store.Close()
	key := batch.Flush.Key
	prior, err := runrecord.LoadOutstandingCheckpointObligations(ctx, store, key)
	if err != nil {
		return err
	}
	accumulator, err := plan.NewBatchAccumulator(*batch.Flush)
	if err != nil {
		return fmt.Errorf("acceptance: %w", err)
	}
	decision := &checkpointPromotion{key: key}
	seeded := map[string]bool{}
	for _, obligation := range prior {
		entry, memoised := g.checkpointMemos[plan.VerificationCheckpoint{ID: obligation.Checkpoint}.GateCheckName()]
		if !memoised || entry.input != obligation.Input {
			g.audit = append(g.audit, fmt.Sprintf("checkpoint obligation stale: key=%s checkpoint=%s obligation=%s; re-accepted by this candidate, not promoted", key, obligation.Checkpoint, obligation.ID))
			continue
		}
		accepted, err := time.Parse(time.RFC3339Nano, obligation.Accepted)
		if err != nil {
			return fmt.Errorf("acceptance: obligation %s: %w", obligation.ID, err)
		}
		if err := decision.add(accumulator, obligation, accepted); err != nil {
			return err
		}
		seeded[obligation.Checkpoint] = true
	}
	for _, checkpoint := range batch.Checkpoints {
		name := checkpoint.GateCheckName()
		if evidence, found := g.terminal[name]; !found || evidence.Outcome != runrecord.LanePassed {
			return fmt.Errorf("acceptance: checkpoint %s lacks passing terminal evidence", checkpoint.ID)
		}
		if seeded[checkpoint.ID] {
			continue
		}
		entry, memoised := g.checkpointMemos[name]
		if !memoised {
			g.audit = append(g.audit, fmt.Sprintf("checkpoint obligation not durable: key=%s checkpoint=%s; memo refused", key, checkpoint.ID))
			continue
		}
		text, err := runrecord.FormatCompletionAcceptanceEvidence(testevidence.CurrentVerifyPolicy, g.planRef, checkpoint.Verify)
		if err != nil {
			return err
		}
		obligation, err := runrecord.NewCheckpointObligation(runrecord.CheckpointObligation{
			Key: key, PlanRef: g.planRef, Checkpoint: checkpoint.ID, Slot: entry.slot, Input: entry.input,
			Evidence: text, Accepted: now.UTC().Format(time.RFC3339Nano), PayloadBytes: int64(len(text)),
		})
		if err != nil {
			return err
		}
		decision.pending = append(decision.pending, obligation)
		if err := decision.add(accumulator, obligation, now); err != nil {
			return err
		}
	}
	if decision.flushed {
		g.audit = append(g.audit, fmt.Sprintf("batch flush: key=%s reason=%s promoted=%d pending=%d", key, decision.reason, len(decision.promoted), len(decision.pending)))
	} else {
		g.audit = append(g.audit, fmt.Sprintf("batch deferred: key=%s outstanding=%d/%d pending=%d; obligations persist for the next candidate", key, accumulator.Pending(key), batch.Flush.MaxSize, len(decision.pending)))
	}
	g.promotion = decision
	return nil
}

// add: one obligation into the accumulator; members after a flush stay
// outstanding for the next batch.
func (decision *checkpointPromotion) add(accumulator *plan.BatchAccumulator, obligation runrecord.CheckpointObligation, at time.Time) error {
	_, flush, reason, err := accumulator.Add(decision.key, obligation.PayloadBytes, at)
	if err != nil {
		return fmt.Errorf("acceptance: %w", err)
	}
	if decision.flushed {
		return nil
	}
	decision.promoted = append(decision.promoted, obligation.ID)
	if flush {
		decision.flushed, decision.reason = true, reason
	}
	return nil
}

// requireFlushedPromotion: parent completion under a flush declaration needs
// the flush; otherwise the accepted checkpoints persist and the row waits.
func (g *gateContext) requireFlushedPromotion() error {
	if g.promotion == nil || g.promotion.flushed {
		return nil
	}
	return fmt.Errorf("acceptance: batch key=%s is not flushed: %d obligations outstanding persist for the next candidate", g.promotion.key, len(g.promotion.promoted))
}

// appendCheckpointObligations: pending obligations ride every final record;
// the promotion rides a successful one only, bound to the gate result.
func (g *gateContext) appendCheckpointObligations(batch *artifact.Batch, result artifact.ID, outcome runrecord.Outcome) error {
	if g.promotion == nil {
		return nil
	}
	for _, obligation := range g.promotion.pending {
		content, err := obligation.Content()
		if err != nil {
			return err
		}
		batch.Contents = append(batch.Contents, content)
	}
	if !g.promotion.flushed || outcome != runrecord.OutcomeSucceeded {
		return nil
	}
	promotion, err := runrecord.NewCheckpointPromotion(runrecord.CheckpointPromotion{
		Key: g.promotion.key, Reason: string(g.promotion.reason), Obligations: g.promotion.promoted, Result: result,
	})
	if err != nil {
		return err
	}
	content, err := promotion.Content()
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, promotion.Lineage()...)
	return nil
}
