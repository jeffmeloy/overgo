package gate

import (
	"fmt"

	"overgo/internal/plan"
)

// sourceProvenMergeGo accepts only Go blobs already present in a protected
// parent. The projected-merge preflight verifies both parents' completion
// authority before this check runs. A conflict resolution that authors new Go
// bytes still needs the row's independent base-failing verifier.
func (g *gateContext) sourceProvenMergeGo() (bool, error) {
	if g.mergeBefore == nil || g.planProjection != plan.MergeProjectionFirstParentTarget || g.mergeSourceStore == "" {
		return false, nil
	}
	parents := g.mergeBefore.parents()
	if len(parents) != 1 {
		return false, fmt.Errorf("merge code proof requires one incoming parent, got %d", len(parents))
	}
	candidate, err := g.plannedTree()
	if err != nil {
		return false, err
	}
	for _, path := range g.plannedGoFiles() {
		digest, exists, err := revisionFileDigest(g.repo, candidate, path)
		if err != nil {
			return false, fmt.Errorf("read merged %s: %w", path, err)
		}
		covered := false
		for _, parent := range []string{g.planHead, parents[0]} {
			parentDigest, parentExists, err := revisionFileDigest(g.repo, parent, path)
			if err != nil {
				return false, fmt.Errorf("read parent %s at %s: %w", path, parent, err)
			}
			if digest == parentDigest && exists == parentExists {
				covered = true
				break
			}
		}
		if !covered {
			return false, nil
		}
	}
	return true, nil
}
