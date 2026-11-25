package main

import (
	"context"
	"fmt"
	"io"
	"slices"
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
	lanes, err := reachedLanes(root, runs, func(base string) ([]string, error) {
		listed, err := gitauthority.Query(ctx, root, "diff", "--name-only", base, "HEAD")
		return strings.Fields(string(listed)), err
	})
	if err != nil {
		return nil, err
	}
	for _, all := range landing.CycleLanes() {
		if !containsLane(lanes, all.Name) {
			fmt.Fprintf(output, "validate -run: cycle lane %s is reached by nothing changed since its pass at %.12s\n", all.Name, runs[all.Name].Passed.Commit)
		}
	}
	suites := leastRecentlyRunFirst(suitesOf(lanes), runs)
	fmt.Fprintf(output, "validate -run: the cycle's lanes run least recently run first: %s\n", strings.Join(laneNames(suites), ", "))
	// A test passed since its row named it is not taken first again (lanesFirst.passedSince): a commit that git cannot
	// read as the other or a descendant of it reads as no pass since.
	passes, after := plan.ReadTestPasses(kept), gitSince(ctx, root)
	// Each lane takes first what the commits since its pass say matters (lanes-first); a lane with no pass kept, or
	// whose commits cannot be read, runs in its own order.
	for index, suite := range suites {
		base := runs[suite.Lane].Passed.Commit
		if base == "" {
			continue
		}
		first, err := readLanesFirst(ctx, root, base, "HEAD")
		if err != nil {
			fmt.Fprintf(output, "validate -run: cycle lane %s runs in its own order: %v\n", suite.Lane, err)
			continue
		}
		left := first.passedSince(passes, after)
		suites[index].First = first
		if len(first.checks)+left > 0 {
			fmt.Fprintf(output, "validate -run: cycle lane %s takes first the %d tests its landed rows' checks name, %d others passed since their row named them\n",
				suite.Lane, len(first.checks), left)
		}
	}
	return suites, nil
}

// leastRecentlyRunFirst orders a close's lanes by when each last ran, passed, failed or yielded (runs' Last), the one
// longest without a run first and one never run before all, the cycle's own order kept among equals (lanes-order).
// A run yields to a newer commit as a lane turns to another package, and every run had begun at the cycle's first lane,
// so the lanes after the models lane, needle-reference and sequences, ran last at 03:39 on 2026-11-24 and not again
// through six closes that day: each run spent its time in the lanes before them and yielded before it reached them. A
// lane that ran goes after the others, so each takes the first place in turn while commits arrive.
func leastRecentlyRunFirst(suites []SelectedSuite, runs map[string]plan.LaneRecord) []SelectedSuite {
	ordered := slices.Clone(suites)
	slices.SortStableFunc(ordered, func(a, b SelectedSuite) int {
		return runs[a.Lane].Last.Ended.Compare(runs[b.Lane].Last.Ended)
	})
	return ordered
}

// laneNames are the suites' lanes, in their order.
func laneNames(suites []SelectedSuite) []string {
	names := make([]string, len(suites))
	for index, suite := range suites {
		names[index] = suite.Lane
	}
	return names
}

// reachedLanes are the cycle's lanes at root a close runs: each cut to what the paths changed since its last pass reach
// (changedSinceLastPass, landing.CycleLanesReached), and a lane run package by package cut further, each package that
// has a pass of its own kept to what the paths changed since that pass reach (lanes-package-pass), its packages read
// together by the pass they share. A lane that failed in one package had kept no pass of the others, so every close ran
// them all against its last pass: the models lane's, at 8dfa48064 on 2026-11-16, three days and every run full-size.
// A package whose own pass diff cannot read keeps the lane's reach.
func reachedLanes(root string, runs map[string]plan.LaneRecord, diff func(base string) ([]string, error)) ([]landing.CycleLane, error) {
	lanes, err := landing.CycleLanesReached(root, changedSinceLastPass(runs, diff))
	if err != nil {
		return nil, err
	}
	emptied := map[string]bool{}
	for index, lane := range lanes {
		own := runs[lane.Name].Packages
		if len(lane.Each) == 0 || len(own) == 0 {
			continue
		}
		bases := map[string]map[string]bool{}
		for _, arguments := range lane.Each {
			pkg := arguments[len(arguments)-1]
			if pass, held := own[pkg]; held && pass.Commit != "" {
				if bases[pass.Commit] == nil {
					bases[pass.Commit] = map[string]bool{}
				}
				bases[pass.Commit][pkg] = true
			}
		}
		dropped := map[string]bool{}
		for base, packages := range bases {
			paths, err := diff(base)
			if err != nil {
				continue
			}
			reached, err := landing.CycleLanesReached(root, map[string][]string{lane.Name: paths})
			if err != nil {
				return nil, err
			}
			enters := map[string]bool{}
			if at := slices.IndexFunc(reached, func(each landing.CycleLane) bool { return each.Name == lane.Name }); at >= 0 {
				for _, arguments := range reached[at].Each {
					enters[arguments[len(arguments)-1]] = true
				}
			}
			for pkg := range packages {
				dropped[pkg] = !enters[pkg]
			}
		}
		lanes[index].Each = slices.DeleteFunc(slices.Clone(lane.Each), func(arguments []string) bool { return dropped[arguments[len(arguments)-1]] })
		emptied[lane.Name] = len(lanes[index].Each) == 0
	}
	return slices.DeleteFunc(lanes, func(lane landing.CycleLane) bool { return emptied[lane.Name] }), nil
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
