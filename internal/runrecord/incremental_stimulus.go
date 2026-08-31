package runrecord

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
)

// VerifyIncrementalStimulus proves a session's cursor delta between two
// admitted stimulus boundaries omits nothing: both boundaries are
// identified, and recomputing the decomposition from their exact admitted
// selections reproduces the delta, so a runtime that receives only the
// delta still observes every admitted stimulus of the current boundary.
func VerifyIncrementalStimulus(
	held, current AttemptStimulusBoundary,
	delta dataset.IncrementalSelection,
) error {
	if held.ID.Kind() != artifact.KindEvidence || current.ID.Kind() != artifact.KindEvidence {
		return errors.New("run record: incremental stimulus requires identified boundaries")
	}
	if held.ID == current.ID {
		return errors.New("run record: incremental stimulus requires distinct boundaries")
	}
	return delta.Validate(held.Selection, current.Selection)
}
