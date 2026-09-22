package gate

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

// mediaReconciliationPrefix and mediaReconciliationSuffix bracket the reviewed
// documents that answer for a move of that runtime.
const (
	mediaReconciliationPrefix = "docs/image_video_"
	mediaReconciliationSuffix = "_reconciliation.json"
)

// mediaRuntimePins is what the merged evidence says about the runtime: the
// paths its identity is taken over, and the identity itself.
type mediaRuntimePins struct {
	RuntimePaths []string `json:"runtime_paths"`
}

// stepSurface refuses a candidate that moves a pinned runtime identity
// without the reconciliation that answers for the move. The media acceptance
// already refuses such a source, but it runs after the commit, so the move is
// found only once it is recorded and has to be undone by a second commit.
// Naming it here costs a file read and reports the path that moved.
func (g *gateContext) stepSurface() (bool, error) {
	data, err := os.ReadFile(filepath.Join(g.repo, filepath.FromSlash(mediaMergedDocument)))
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
	moved := movedRuntimePaths(g.paths, pins.RuntimePaths)
	if len(moved) == 0 || reconciliationShips(g.paths) {
		return len(moved) == 0, nil
	}
	return false, fmt.Errorf(
		"moves the pinned media runtime without its reconciliation: %s; land the reviewed reconciliation in this commit, or keep the change outside %s",
		strings.Join(moved, ", "), strings.Join(pins.RuntimePaths, ", "),
	)
}

// movedRuntimePaths names the shipped paths that lie under a pinned runtime
// path, in the order the candidate ships them.
func movedRuntimePaths(shipped, pinned []string) []string {
	var moved []string
	for _, path := range shipped {
		clean := filepath.ToSlash(path)
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
// evidence itself or a reviewed reconciliation document, either of which
// answers for a move.
func reconciliationShips(shipped []string) bool {
	for _, path := range shipped {
		clean := filepath.ToSlash(path)
		if clean == mediaMergedDocument {
			return true
		}
		if strings.HasPrefix(clean, mediaReconciliationPrefix) && strings.HasSuffix(clean, mediaReconciliationSuffix) {
			return true
		}
	}
	return false
}
