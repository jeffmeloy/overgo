package plan

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/worklease"
)

// LandedRowTrailer marks the commit that landed a row; the loop confirms a
// landing and reads progress from it.
const LandedRowTrailer = "Landed-Row"

// AcceptedProgressSince reports whether a landing on the first-parent chain
// since the checkpoint moved an open row forward: a sibling step of its item,
// or a prerequisite it waits on, landed. Git holds the record: each landing
// names its row in its Landed-Row trailer.
func AcceptedProgressSince(ctx context.Context, repository string, document Plan, itemID, stepID, checkpoint string) (bool, error) {
	if ctx == nil || !worklease.ValidCommit(checkpoint) {
		return false, errors.New("plan: progress requires an exact checkpoint")
	}
	step, found := exactPlanStep(document, itemID, stepID)
	if !found || step.Status != StatusOpen {
		return false, errors.New("plan: progress step is not open")
	}
	log, err := gitCommand(ctx, repository, "log", "--first-parent", "--no-merges", "--format=%B%x00", checkpoint+"..HEAD")
	if err != nil {
		return false, err
	}
	for message := range strings.SplitSeq(string(log), "\x00") {
		for line := range strings.Lines(message) {
			landed, found := strings.CutPrefix(strings.TrimSpace(line), LandedRowTrailer+": ")
			if found && (strings.HasPrefix(landed, itemID+"/") || slices.Contains(step.DependsOn, landed)) {
				return true, nil
			}
		}
	}
	return false, nil
}
