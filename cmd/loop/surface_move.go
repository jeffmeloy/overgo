package main

import (
	"context"
	"fmt"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/inferencesurface"
)

// surfaceMoveShown bounds how many moved sources a warning names; the count
// says how many there are.
const surfaceMoveShown = 6

// renameArrow separates the old path from the new one in a porcelain line.
const renameArrow = " -> "

// porcelainPathOffset is where the path starts in a porcelain status line,
// after its two status letters and a space.
const porcelainPathOffset = 3

// shipSetPaths reads the paths of a porcelain status: the ship set of a
// landing is the dirty tree. A rename names both paths, and both count.
func shipSetPaths(porcelain string) []string {
	var paths []string
	for line := range strings.Lines(porcelain) {
		line = strings.TrimRight(line, "\r\n")
		if len(line) <= porcelainPathOffset {
			continue
		}
		for path := range strings.SplitSeq(line[porcelainPathOffset:], renameArrow) {
			paths = append(paths, strings.Trim(path, `"`))
		}
	}
	return paths
}

// surfaceMoveWarning says, before the gate, that the ship set moves the
// inference surface. Validation evidence is keyed to that surface, so such a
// landing expires the long-form guard record of every model; it was learnt
// only from a review candidate after the commit, and acknowledged there
// without being weighed. It is a warning and not a refusal: moving the surface
// is sometimes the work, and the worker who reads this decides to batch it.
func surfaceMoveWarning(moves []string) string {
	if len(moves) == 0 {
		return ""
	}
	shown := moves[:min(len(moves), surfaceMoveShown)]
	return fmt.Sprintf("advisory: warning: this landing moves the inference surface (%d source(s): %s): every long-form guard record keyed to it expires; batch such changes and re-acquire the guards once at the settled surface",
		len(moves), strings.Join(shown, ", "))
}

// inferenceSurfaceMoves asks the surface's owner which paths of the dirty
// tree move it.
func inferenceSurfaceMoves(ctx context.Context, root string) ([]string, error) {
	porcelain, err := gitauthority.Query(ctx, root, "status", "--porcelain", "--untracked-files=all")
	if err != nil {
		return nil, err
	}
	return inferencesurface.Moves(ctx, root, shipSetPaths(string(porcelain)))
}
