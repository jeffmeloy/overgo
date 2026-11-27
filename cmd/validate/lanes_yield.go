package main

import (
	"context"
	"os"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
)

// runnerYield is the yield of a run a runner starts to the commit its due file names (lanes-first/yield): the run's own
// commit, the due file, and the reading of the run's own commit. A yield loses nothing: each test the run ran kept its
// result as it ended, and the run at the newer commit takes what has none current (lanes-record).
type runnerYield struct {
	own, due string
	headErr  error
}

// yieldTo is the yield of a run at root to the commit the due file at due names.
func yieldTo(ctx context.Context, root, due string) runnerYield {
	head, err := gitauthority.Query(ctx, root, "rev-parse", "HEAD")
	return runnerYield{own: strings.TrimSpace(string(head)), due: due, headErr: err}
}

// yields is the commit the due file names, where it is another than the run's own, which the runner names due as a
// newer close lands.
func (y runnerYield) yields() (string, bool) {
	var named struct {
		Commit string `json:"commit"`
	}
	if y.headErr != nil || y.own == "" || jsonfile.Decode(y.due, &named) != nil || named.Commit == "" || named.Commit == y.own {
		return "", false
	}
	return named.Commit, true
}

// lanesYield is the yield of the run a runner started, which names its due file in plan.LanesDueEnvironment
// (yieldTo); a run no runner started runs to its end, given none.
func lanesYield(ctx context.Context, root string) func() (string, bool) {
	due := os.Getenv(plan.LanesDueEnvironment)
	if due == "" {
		return nil
	}
	return yieldTo(ctx, root, due).yields
}
