package longform

import (
	"context"
	"strings"

	"overgo/internal/inferencesurface"
)

// Affected reports whether changed paths can alter what a model computes, by
// asking the inference surface's owner the same question that keys retained
// evidence: a path that moves the surface needs measurement, and one that
// does not leaves every record keyed to it current. Unknown scope requires
// measurement. The caller still owns corpus and model-input identity
// validation.
func Affected(ctx context.Context, root string, paths []string) (bool, string, error) {
	if len(paths) == 0 {
		return true, "unknown changed-path scope; full selected catalog required", nil
	}
	moves, err := inferencesurface.Moves(ctx, root, paths)
	if err != nil {
		return false, "", err
	}
	if len(moves) != 0 {
		return true, "the change moves the inference surface (" + strings.Join(moves, ", ") + "); full selected catalog required", nil
	}
	return false, "the change leaves the inference surface unchanged; no model measurements required", nil
}
