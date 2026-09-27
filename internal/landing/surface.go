package landing

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// mediaMergedDocument names the media evidence that pins the runtime its
// generation claims were acquired on.
const mediaMergedDocument = "docs/image_video_merged.json"

// mediaReviewedDeltas names the one registry of reviewed output-neutral
// deltas that answers for a move of that runtime.
const mediaReviewedDeltas = "docs/media_reviewed_deltas.json"

// mediaRuntimePins is what the merged evidence says about the runtime: the
// paths its identity is taken over.
type mediaRuntimePins struct {
	RuntimePaths []string `json:"runtime_paths"`
}

// MediaMoves refuses shipped paths that move the pinned media runtime
// without the reconciliation that answers for the move; the media acceptance
// finds such a move only after the commit. It reports whether no pinned path
// moved.
func MediaMoves(repo string, paths []string) (bool, error) {
	data, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(mediaMergedDocument)))
	if errors.Is(err, fs.ErrNotExist) {
		return true, nil
	}
	if err != nil {
		return false, err
	}
	var pins mediaRuntimePins
	if err := json.Unmarshal(data, &pins); err != nil {
		return false, fmt.Errorf("%s: %w", mediaMergedDocument, err)
	}
	if len(pins.RuntimePaths) == 0 {
		return false, fmt.Errorf("%s names no runtime paths", mediaMergedDocument)
	}
	moved := movedRuntimePaths(paths, pins.RuntimePaths)
	if len(moved) == 0 || reconciliationShips(paths) {
		return len(moved) == 0, nil
	}
	return false, fmt.Errorf(
		"moves the pinned media runtime without its reconciliation: %s; land its reviewed delta in %s in this commit, or keep the change outside %s",
		strings.Join(moved, ", "), mediaReviewedDeltas, strings.Join(pins.RuntimePaths, ", "),
	)
}

// movedRuntimePaths names the shipped paths that lie under a pinned runtime
// path, in the order the candidate ships them. A test file is not runtime
// source: the pinned identity is taken over production files only.
func movedRuntimePaths(shipped, pinned []string) []string {
	var moved []string
	for _, path := range shipped {
		clean := filepath.ToSlash(path)
		if strings.HasSuffix(clean, "_test.go") {
			continue
		}
		for _, pin := range pinned {
			if clean == pin || strings.HasPrefix(clean, pin+"/") {
				moved = append(moved, clean)
				break
			}
		}
	}
	return moved
}

// reconciliationShips reports whether the candidate carries the merged
// evidence itself or the reviewed delta registry, either of which answers
// for a move.
func reconciliationShips(shipped []string) bool {
	for _, path := range shipped {
		if clean := filepath.ToSlash(path); clean == mediaMergedDocument || clean == mediaReviewedDeltas {
			return true
		}
	}
	return false
}
