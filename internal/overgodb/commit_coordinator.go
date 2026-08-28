package overgodb

import (
	"crypto/sha256"
	"fmt"

	"overgo/internal/artifact"
)

// commitCoordinator owns the one ordered transaction path every
// storage module rides, on the write side and on replay: idempotence,
// head precondition, facet validation, delta reduction, the canonical
// fixed point, encoding, durable append, and facet application. A
// failure before the append publishes nothing; an append error is a
// store fault the owner must record before refusing further writes; a
// successful append advances every facet exactly once.
type commitCoordinator struct {
	state *catalogState
	log   *recordLog
}

// commitAdvance is the head movement a successful commit produces.
type commitAdvance struct {
	id        artifact.CommitID
	sequence  uint64
	replayEnd int64
}

// appendFault wraps an error raised during the durable append, where
// the journal state is uncertain: the store must fault and reopen.
type appendFault struct{ cause error }

// Error reports the underlying append failure.
func (f appendFault) Error() string { return f.cause.Error() }

// Unwrap exposes the cause for errors.Is inspection.
func (f appendFault) Unwrap() error { return f.cause }

// commit executes one write transaction against the given head. The
// replayed result reports an exact idempotent repeat of an already
// committed key; every other early return leaves state untouched.
func (c commitCoordinator) commit(
	normalized artifact.Batch,
	payloadHash [sha256.Size]byte,
	head artifact.CommitID,
	sequence uint64,
) (advance commitAdvance, replayed bool, err error) {
	if committed, ok := c.state.commit(normalized.Key); ok {
		if committed.payload == payloadHash {
			return commitAdvance{id: committed.id}, true, nil
		}
		return commitAdvance{}, false, fmt.Errorf("%w: %q", ErrBatchKeyConflict, normalized.Key)
	}
	if normalized.ExpectedHead != nil && *normalized.ExpectedHead != head {
		return commitAdvance{}, false, fmt.Errorf("%w: expected %s, have %s", ErrHeadConflict, *normalized.ExpectedHead, head)
	}
	if err := c.state.validate(normalized); err != nil {
		return commitAdvance{}, false, err
	}
	delta := c.state.delta(normalized)
	if delta.Empty() {
		return commitAdvance{}, false, ErrNoChange
	}
	// The persisted delta must round-trip the replay-side canonical
	// check, which re-normalizes it: normalization re-derives manifest
	// descriptors and lineage, so a delta whose manifest artifacts were
	// deduplicated against existing state would re-expand on replay and
	// never match its own bytes. Normalizing once more before persisting
	// makes the check a fixed point.
	delta, err = normalizeBatch(delta)
	if err != nil {
		return commitAdvance{}, false, err
	}
	payload, locators, err := encodeTransaction(payloadHash, delta)
	if err != nil {
		return commitAdvance{}, false, err
	}
	next := sequence + 1
	id, payloadOffset, replayEnd, err := c.log.append(next, head, payload)
	if err != nil {
		return commitAdvance{id: id}, false, appendFault{cause: err}
	}
	bindContentLocators(locators, payloadOffset)
	c.state.apply(delta, locators, next)
	c.state.addCommit(committedBatch{key: normalized.Key, id: id, payload: payloadHash, sequence: next})
	return commitAdvance{id: id, sequence: next, replayEnd: replayEnd}, false, nil
}

// replay applies one already-durable journal record through the same
// validation and application transitions the write path uses.
func (c commitCoordinator) replay(record logRecord) error {
	batch, payloadHash, locators, err := decodeRecord(record)
	if err != nil {
		return err
	}
	if batch.ExpectedHead != nil && *batch.ExpectedHead != record.previous {
		return fmt.Errorf("%w: recorded predecessor differs", ErrHeadConflict)
	}
	if _, exists := c.state.commit(batch.Key); exists {
		return fmt.Errorf("%w: %q repeats in log", ErrBatchKeyConflict, batch.Key)
	}
	if err := c.state.validate(batch); err != nil {
		return err
	}
	c.state.apply(batch, locators, record.sequence)
	c.state.addCommit(committedBatch{key: batch.Key, id: record.id, payload: payloadHash, sequence: record.sequence})
	return nil
}
