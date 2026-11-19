package main

import (
	"context"
	"fmt"
	"io"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/landing"
	"overgo/internal/plan"
)

// changedSinceLastPass names, for each lane runs keeps a pass of, the paths changed since that pass (diff), which the
// lane's reach is read from (landing.CycleLanesReached); a lane with no pass kept, or whose pass's commit diff cannot
// read, is left out and runs whole.
func changedSinceLastPass(runs map[string]plan.LaneRecord, diff func(base string) ([]string, error)) map[string][]string {
	changed := map[string][]string{}
	for name, record := range runs {
		if record.Passed.Commit == "" {
			continue
		}
		if paths, err := diff(record.Passed.Commit); err == nil {
			changed[name] = paths
		}
	}
	return changed
}

// reachedSuites are the cycle's lanes as a run on a row's close runs them (lanes-reach): each cut to what the paths
// changed at root since its last pass, kept at kept, reach; it says on output each lane it leaves out.
func reachedSuites(ctx context.Context, output io.Writer, root, kept string) ([]SelectedSuite, error) {
	runs := plan.ReadLaneRuns(kept)
	changed := changedSinceLastPass(runs, func(base string) ([]string, error) {
		listed, err := gitauthority.Query(ctx, root, "diff", "--name-only", base, "HEAD")
		return strings.Fields(string(listed)), err
	})
	lanes, err := landing.CycleLanesReached(root, changed)
	if err != nil {
		return nil, err
	}
	for _, all := range landing.CycleLanes() {
		if !containsLane(lanes, all.Name) {
			fmt.Fprintf(output, "validate -run: cycle lane %s is reached by nothing changed since its pass at %.12s\n", all.Name, runs[all.Name].Passed.Commit)
		}
	}
	return suitesOf(lanes), nil
}

// containsLane says whether lanes holds the lane named.
func containsLane(lanes []landing.CycleLane, name string) bool {
	for _, lane := range lanes {
		if lane.Name == name {
			return true
		}
	}
	return false
}
