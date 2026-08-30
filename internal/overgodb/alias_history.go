package overgodb

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
)

// AliasEventRange bounds a validated alias-history read to committed journal
// coordinates. Both endpoints are inclusive and ToSequence must name a commit
// already visible through the Store handle. A zero FromSequence starts at the
// beginning of the journal.
type AliasEventRange struct {
	Prefix       string
	FromSequence uint64
	ToSequence   uint64
}

// AliasEvent is one exact alias delta and the commit that made it durable.
type AliasEvent struct {
	Commit  CommitView
	Binding artifact.AliasBinding
}

// VisitAliasEvents replays the validated journal range and visits its exact
// alias deltas in commit order. It exposes history without adding mutable
// alias chronology to the catalog projections or their checkpoints.
func (s *Store) VisitAliasEvents(
	ctx context.Context,
	eventRange AliasEventRange,
	visit func(AliasEvent) error,
) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if eventRange.ToSequence == 0 || eventRange.ToSequence < eventRange.FromSequence {
		return errors.New("overgodb: invalid alias event range")
	}
	if visit == nil {
		return errors.New("overgodb: nil alias event visitor")
	}
	s.mu.RLock()
	if err := s.ready(false); err != nil {
		s.mu.RUnlock()
		return err
	}
	if eventRange.ToSequence > s.sequence {
		s.mu.RUnlock()
		return errors.New("overgodb: alias event range exceeds store head")
	}
	root := s.root
	expected := s.state.commits.at(int(eventRange.ToSequence - 1)).id
	s.mu.RUnlock()

	var observed artifact.CommitID
	log, _, err := openRecordLog(root, true, replayAnchor{}, func(record logRecord) error {
		if record.sequence == eventRange.ToSequence {
			observed = record.id
		}
		if record.sequence < eventRange.FromSequence || record.sequence > eventRange.ToSequence {
			return nil
		}
		if err := contextError(ctx); err != nil {
			return err
		}
		batch, _, _, err := decodeRecord(record)
		if err != nil {
			return err
		}
		commit := CommitView{Key: batch.Key, ID: record.id, Sequence: record.sequence}
		for _, binding := range artifact.CloneAliasBindings(batch.Aliases) {
			if strings.HasPrefix(binding.Name, eventRange.Prefix) {
				if err := visit(AliasEvent{Commit: commit, Binding: binding}); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if log != nil {
		err = errors.Join(err, log.Close())
	}
	if err != nil {
		return fmt.Errorf("overgodb: visit alias events: %w", err)
	}
	if observed != expected {
		return errors.New("overgodb: alias event range head differs from store view")
	}
	return nil
}
