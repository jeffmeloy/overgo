package gate

import (
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"slices"

	"overgo/internal/jsonfile"
	"overgo/internal/plan"
)

// admitProofHorizon holds a landing that moves the proof horizon to the one
// value this gate can vouch for: the revision it has just proved, under the
// horizon already committed, with the completions it counted. Everything at
// or before that revision was therefore proved from the store at least once,
// by a gate, before Git alone is believed for it. Removing the receipt asks
// for more proof and is always admitted.
func (g *gateContext) admitProofHorizon(proved plan.CompletionAuthority) error {
	if !slices.Contains(g.paths, plan.ProofHorizonPath) {
		return nil
	}
	var shipped plan.ProofHorizon
	err := jsonfile.DecodeStrict(filepath.Join(g.repo, filepath.FromSlash(plan.ProofHorizonPath)), &shipped)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("commit admission: proof horizon: %w", err)
	}
	if sealed := proved.SealProofHorizon(); shipped != sealed {
		return fmt.Errorf(
			"commit admission: the proof horizon moves only to the revision this gate proved (%.12s, %d completions), not to %.12s with %d; run `go run ./cmd/plan -seal-horizon`",
			sealed.Commit, sealed.Completions, shipped.Commit, shipped.Completions,
		)
	}
	return nil
}
