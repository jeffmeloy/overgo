package overgodb

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

// RecentArtifact is one artifact of a kind as the store introduced it,
// with the commit sequence that introduced it, whether its payload is
// held inline, and the runs that produced it.
type RecentArtifact struct {
	Descriptor artifact.Descriptor
	Sequence   uint64
	Payload    bool
	Producers  []artifact.ID
}

// RecentArtifacts walks a kind's artifacts from the newest introduction
// backwards and returns the first limit that keep admits (every one when
// keep is nil), newest first. A query pages by identity, so this is the
// one read that answers "the latest files"; the walk is bounded by the
// kind's count and reads the in-memory catalog only.
func (s *Store) RecentArtifacts(ctx context.Context, kind artifact.Kind, limit int, keep func(artifact.Descriptor) bool) ([]RecentArtifact, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	if limit <= 0 {
		return nil, errors.New("overgodb: recent artifacts need a positive limit")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return nil, err
	}
	ids := s.state.artifacts.byKind[kind]
	recent := make([]RecentArtifact, 0, min(limit, len(ids)))
	for index := len(ids); index > 0 && len(recent) < limit; {
		index--
		record, ok := s.state.artifacts.record(ids[index])
		if !ok || keep != nil && !keep(record.descriptor) {
			continue
		}
		producers := make([]artifact.ID, 0)
		for _, edge := range s.state.lineage.parentsOf(ids[index]) {
			if edge.relation == artifact.RelationProducedBy {
				producers = append(producers, edge.parent)
			}
		}
		recent = append(recent, RecentArtifact{Descriptor: record.descriptor, Sequence: record.sequence, Payload: s.state.contents.has(ids[index]), Producers: producers})
	}
	return recent, nil
}
