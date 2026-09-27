package gate

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/inferencesurface"
	"overgo/internal/landing"
)

// surfaceMoveShown bounds how many moved sources the warning names; the count
// says how many there are.
const surfaceMoveShown = 6

// stepSurface refuses a candidate that moves a pinned runtime identity
// without the reconciliation that answers for the move. The media acceptance
// already refuses such a source, but it runs after the commit, so the move is
// found only once it is recorded and has to be undone by a second commit.
// Naming it here costs a file read and reports the path that moved.
// A change to bytes a record pins is found here for the same reason. A move
// of the inference surface is advised, not refused: validation evidence is
// keyed to that surface, so the move expires every model's long-form guard
// record, and moving it is sometimes the work -- the worker batches such
// changes. A landing that cannot tell whether it moves the surface still lands.
func (g *gateContext) stepSurface() (bool, error) {
	if moves, err := inferencesurface.Moves(context.Background(), g.repo, g.paths); err == nil && len(moves) != 0 {
		g.advise(noteWarning, fmt.Sprintf("this landing moves the inference surface (%d source(s): %s): every long-form guard record keyed to it expires; batch such changes and re-acquire the guards once at the settled surface",
			len(moves), strings.Join(moves[:min(len(moves), surfaceMoveShown)], ", ")))
	}
	quiet, err := landing.MediaMoves(g.repo, g.paths)
	stale, pinErr := landing.StalePins(g.repo, g.paths)
	if pinErr == nil && len(stale) != 0 {
		pinErr = fmt.Errorf("changes bytes a record pins: %s; repin the record for a behaviour-neutral edit, or restore the file", strings.Join(stale, "; "))
	}
	return quiet && len(stale) == 0, errors.Join(err, pinErr)
}
