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

// stillCurrent says whether a result kept under key, a test of pkg in lane, or the lane itself run whole (pkg empty),
// is a pass that is current: one whose commit's changes, as diff reads them, reach neither the package nor, for a lane
// run whole, the lane (landing.CycleLanesReached), each commit's changes read once a lane. A failure, a test with no
// result, and a commit whose changes diff cannot read are never current (lanes-record): a slow test runs again only when
// what it reads changed since it passed.
func stillCurrent(root string, results map[string]plan.TestResult, diff func(base string) ([]string, error)) func(lane, pkg, key string) bool {
	reached := map[[2]string]map[string]bool{}
	return func(lane, pkg, key string) bool {
		result, held := results[key]
		if !held || result.Failed || result.Commit == "" {
			return false
		}
		at := [2]string{result.Commit, lane}
		packages, read := reached[at]
		if !read {
			paths, err := diff(result.Commit)
			lanes, reachErr := landing.CycleLanesReached(root, map[string][]string{lane: paths})
			if err == nil && reachErr == nil {
				packages = map[string]bool{}
				// A lane the changes are not given for is returned whole, so only the lane asked about is read.
				for _, each := range lanes {
					if each.Name != lane {
						continue
					}
					packages[""] = true
					for _, arguments := range each.Each {
						packages[arguments[len(arguments)-1]] = true
					}
				}
			}
			reached[at] = packages
		}
		return packages != nil && !packages[pkg]
	}
}

// testKey is how a test of a lane's package, as its arguments name the package, is kept (plan.PassedTest), and, with no
// package, how a lane run whole named test is kept (plan.LaneTest).
func testKey(pkg, test string) string {
	if pkg == "" {
		return plan.LaneTest(test)
	}
	return plan.PassedTest(modulePrefix+strings.TrimPrefix(pkg, "./"), test)
}

// reachedSuites are the cycle's lanes as a run on a row's close runs them (lanes-record): a lane run test by test takes
// each test whose last result kept at kept is not a pass current at root's HEAD, and a lane run whole runs unless its own
// is; it says on output each lane it leaves out.
func reachedSuites(ctx context.Context, output io.Writer, root, kept string) ([]SelectedSuite, error) {
	current := stillCurrent(root, plan.ReadTestResults(kept), func(base string) ([]string, error) {
		listed, err := gitauthority.Query(ctx, root, "diff", "--name-only", base, "HEAD")
		return strings.Fields(string(listed)), err
	})
	var suites []SelectedSuite
	for _, suite := range cycleSuites() {
		lane := suite.Lane
		if len(suite.Each) == 0 && current(lane, "", testKey("", lane)) {
			fmt.Fprintf(output, "validate -run: cycle lane %s passed at a commit since which nothing it reads changed\n", lane)
			continue
		}
		suite.Current = func(pkg, test string) bool { return current(lane, pkg, testKey(pkg, test)) }
		suites = append(suites, suite)
	}
	return suites, nil
}
