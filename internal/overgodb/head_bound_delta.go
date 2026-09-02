package overgodb

import (
	"context"
	"crypto/sha256"
	"fmt"
	"slices"

	"overgo/internal/artifact"
)

// HeadBoundDelta is the ordered, coalesced view change between two exact
// store heads. Artifact, content, and lineage additions are append-only and
// therefore commute within the window; alias changes coalesce to each
// touched name's final state in commit order, which is the only coalescing
// an ordered window admits. Anything else — retention, rebuild, a projection
// contract change — breaks continuity and forces a bounded snapshot resync
// instead of a delta.
type HeadBoundDelta struct {
	PreviousHead      artifact.CommitID     `json:"previous_head"`
	PreviousSequence  uint64                `json:"previous_sequence"`
	Head              artifact.CommitID     `json:"head"`
	Sequence          uint64                `json:"sequence"`
	ProjectionVersion string                `json:"projection_version"`
	Artifacts         []artifact.Descriptor `json:"artifacts,omitempty"`
	Contents          []artifact.ID         `json:"contents,omitempty"`
	Aliases           []AliasView           `json:"aliases,omitempty"`
	RemovedAliases    []string              `json:"removed_aliases,omitempty"`
	Lineage           []artifact.Lineage    `json:"lineage,omitempty"`
	Commits           []CommitView          `json:"commits,omitempty"`
	Truncated         bool                  `json:"truncated,omitzero"`
}

// ProjectionContractVersion digests the closed projection table — every
// projection name and version — so a consumer holding view state derived
// under one contract can detect that a rebuilt or upgraded store no longer
// speaks it.
func ProjectionContractVersion() string {
	state := newCatalogState()
	hasher := sha256.New()
	for _, projection := range projections(&state) {
		fmt.Fprintf(hasher, "%s:%d\x00", projection.name, projection.version)
	}
	return fmt.Sprintf("%x", hasher.Sum(nil)[:12])
}

// DeltasSince serves the coalesced view change from a consumer's exact last
// head to the current head, bounded to maxCommits journal commits per call.
// resync reports that no delta can bridge the consumer's state —
// an unknown or diverged previous coordinate, a head behind a retention or
// rebuild, or a differing projection contract — and the consumer must take
// a bounded snapshot resync instead of applying anything.
func (s *Store) DeltasSince(
	ctx context.Context,
	previousHead artifact.CommitID,
	previousSequence uint64,
	projectionVersion string,
	maxCommits int,
) (HeadBoundDelta, bool, error) {
	if maxCommits <= 0 {
		return HeadBoundDelta{}, false, fmt.Errorf("overgodb: head-bound delta window must be positive")
	}
	if err := contextError(ctx); err != nil {
		return HeadBoundDelta{}, false, err
	}
	if projectionVersion != ProjectionContractVersion() {
		return HeadBoundDelta{}, true, nil
	}
	_, sequence := s.Head()
	if previousSequence > sequence {
		return HeadBoundDelta{}, true, nil
	}
	if previousSequence == 0 {
		// A consumer with no state starts from a snapshot, never a delta.
		return HeadBoundDelta{}, true, nil
	}
	anchor, found, err := s.CommitAt(ctx, previousSequence)
	if err != nil {
		return HeadBoundDelta{}, false, err
	}
	if !found || anchor.ID != previousHead {
		return HeadBoundDelta{}, true, nil
	}
	delta := HeadBoundDelta{
		PreviousHead: previousHead, PreviousSequence: previousSequence,
		Head: previousHead, Sequence: previousSequence,
		ProjectionVersion: projectionVersion,
	}
	if previousSequence == sequence {
		return delta, false, nil
	}
	last := previousSequence + uint64(maxCommits)
	if last > sequence {
		last = sequence
	} else if last < sequence {
		delta.Truncated = true
	}
	seenArtifacts := map[artifact.ID]struct{}{}
	finalAliases := map[string]artifact.AliasBinding{}
	aliasOrder := []string{}
	for next := previousSequence + 1; next <= last; next++ {
		commit, found, err := s.commitDeltaAt(ctx, next, false)
		if err != nil {
			return HeadBoundDelta{}, false, err
		}
		if !found || commit.Previous != delta.Head {
			// The journal moved underneath the walk: hand the consumer a
			// resync rather than a delta with a hole in it.
			return HeadBoundDelta{}, true, nil
		}
		for _, descriptor := range commit.Delta.Artifacts {
			if _, seen := seenArtifacts[descriptor.ID]; seen {
				continue
			}
			seenArtifacts[descriptor.ID] = struct{}{}
			delta.Artifacts = append(delta.Artifacts, descriptor)
		}
		for _, content := range commit.Delta.Contents {
			if _, seen := seenArtifacts[content.Descriptor.ID]; !seen {
				seenArtifacts[content.Descriptor.ID] = struct{}{}
				delta.Artifacts = append(delta.Artifacts, content.Descriptor)
			}
			delta.Contents = append(delta.Contents, content.Descriptor.ID)
		}
		for _, binding := range commit.Delta.Aliases {
			if _, touched := finalAliases[binding.Name]; !touched {
				aliasOrder = append(aliasOrder, binding.Name)
			}
			finalAliases[binding.Name] = binding
		}
		delta.Lineage = append(delta.Lineage, commit.Delta.Lineage...)
		delta.Commits = append(delta.Commits, commit.Commit)
		delta.Head, delta.Sequence = commit.Commit.ID, commit.Commit.Sequence
	}
	slices.Sort(aliasOrder)
	for _, name := range aliasOrder {
		binding := finalAliases[name]
		if binding.Remove {
			delta.RemovedAliases = append(delta.RemovedAliases, name)
			continue
		}
		delta.Aliases = append(delta.Aliases, AliasView{Name: name, Target: binding.Target})
	}
	return delta, false, nil
}
