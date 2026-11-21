package overgodb

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// OutsideArtifact is an artifact the store records a local location of outside a set of roots with no live copy inside
// them and no content of its own: its outside locations that exist (Live), which removing those folders would lose, and
// those that do not (Gone).
type OutsideArtifact struct {
	Artifact artifact.ID
	Live     []string
	Gone     []string
}

// Lost says no copy of the artifact is left: none of its recorded locations exists and the store holds no content.
func (a OutsideArtifact) Lost() bool {
	return len(a.Live) == 0
}

// OutsideLocations lists the artifacts the store records a local file or directory of outside roots (the repository
// and the data roots), with no live copy inside them and no content the store holds: each whose live copies lie only
// outside, and each with none left (store-roots, the owner's review of 2026-11-20: a test's data lies in the repository
// or the data root, and a capture whose only copy sat in a removed worktree's tmp was lost with it). Listed by their
// first location.
func (s *Store) OutsideLocations(ctx context.Context, roots []string) ([]OutsideArtifact, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return nil, err
	}
	var listed []OutsideArtifact
	for id, locations := range s.state.locations.byArtifact {
		entry, outside, kept := OutsideArtifact{Artifact: id}, false, s.state.contents.has(id)
		for _, location := range locations {
			if location.Kind != artifact.LocationFile && location.Kind != artifact.LocationDirectory {
				continue
			}
			_, err := os.Stat(location.Value)
			inside := withinRoots(roots, location.Value)
			switch {
			case err == nil && inside:
				kept = true
			case err == nil:
				entry.Live = append(entry.Live, location.Value)
			case !inside:
				entry.Gone = append(entry.Gone, location.Value)
			}
			outside = outside || !inside
		}
		if outside && !kept {
			listed = append(listed, entry)
		}
	}
	slices.SortFunc(listed, func(a, b OutsideArtifact) int {
		return strings.Compare(slices.Concat(a.Live, a.Gone)[0], slices.Concat(b.Live, b.Gone)[0])
	})
	return listed, nil
}

// withinRoots says path lies in one of roots.
func withinRoots(roots []string, path string) bool {
	for _, root := range roots {
		if relative, err := filepath.Rel(root, path); err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative) {
			return true
		}
	}
	return false
}
