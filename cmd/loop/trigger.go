package main

import (
	"bytes"
	"context"
	"fmt"
	"path"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/scorecard"
)

// A cycle measures what the commands its cells run execute, so a commit moves what it
// measures only when it changes a Go package in the closure of one of them, or a file the
// kernels live in. The closure is what go list reports, not a list kept by hand.
var (
	// surfaceCommands are the commands a cycle's cells run.
	surfaceCommands = []string{"./cmd/evaluate", "./cmd/train", "./cmd/longform", "./cmd/recipe", "./cmd/server", "./cmd/validate"}
	// surfacePrefixes are the paths outside Go packages a cell's computation reads.
	surfacePrefixes = []string{"kernels/"}
)

// surfaceClosure is the repository's packages, as directories, a cycle's commands depend on.
func surfaceClosure(ctx context.Context, root string) (map[string]bool, error) {
	var listed, diagnostics bytes.Buffer
	command := append([]string{"go", "-C", root, "list", "-deps", "-f", "{{.ImportPath}}"}, surfaceCommands...)
	if err := runToolEnv(loopEnvironment(), nil, &listed, &diagnostics, command...); err != nil {
		return nil, fmt.Errorf("go list the cycle's commands: %w: %s", err, strings.TrimSpace(diagnostics.String()))
	}
	closure := map[string]bool{}
	for line := range strings.SplitSeq(listed.String(), "\n") {
		if directory, found := strings.CutPrefix(strings.TrimSpace(line), "overgo/"); found {
			closure[directory] = true
		}
	}
	return closure, nil
}

// touchesSurface reports whether a changed file moves what a cycle measures: a non-test Go file of a
// package in the closure, or a path under a surface prefix.
func touchesSurface(file string, closure map[string]bool) bool {
	switch {
	case slices.ContainsFunc(surfacePrefixes, func(prefix string) bool { return strings.HasPrefix(file, prefix) }):
		return true
	case !strings.HasSuffix(file, ".go") || strings.HasSuffix(file, "_test.go") || strings.Contains(file, "/testdata/"):
		return false
	}
	return closure[path.Dir(file)]
}

// commitAbbreviation is how many hex digits of a commit a decision's reason shows.
const commitAbbreviation = 12

// cycleDecision is whether a cycle is due and why, in words an operator reads.
type cycleDecision struct {
	Due    bool
	Reason string
}

// decideCycle says whether an improvement cycle is due from the landings since the last cycle
// that completed, of any outcome: none is due when nothing a cycle measures moved, so a failing
// cycle is not started again until the surface changes; one is due once landings that moved it
// number at least limit, the count at which finding which of them broke a failed cycle would cost
// more than the cycle. A moved surface below the limit waits for the plan to empty.
func decideCycle(last string, moved, limit int) cycleDecision {
	switch {
	case last == "":
		return cycleDecision{Reason: "no cycle has completed, so there is no run to compare with"}
	case moved == 0:
		return cycleDecision{Reason: "nothing a cycle measures has changed since the cycle at " + last[:min(len(last), commitAbbreviation)]}
	case limit > 0 && moved >= limit:
		return cycleDecision{Due: true, Reason: "landings that moved what a cycle measures reached the limit"}
	}
	return cycleDecision{Reason: "what a cycle measures moved, but the landings are below the limit, so it waits for the plan to empty"}
}

// landingsThatMovedTheSurface counts the commits after the newest published scorecard's that touch
// what a cycle measures, and reports that scorecard's commit.
func landingsThatMovedTheSurface(ctx context.Context, root, records string, closure map[string]bool) (last string, moved int, err error) {
	published, err := scorecard.Records(records)
	if err != nil || len(published) == 0 {
		return "", 0, err
	}
	last = published[len(published)-1].Commit
	log, err := gitauthority.Query(ctx, root, "log", "--name-only", "--format=%x00%H", last+"..HEAD")
	if err != nil {
		return last, 0, err
	}
	for commit := range strings.SplitSeq(string(log), "\x00") {
		_, files, _ := strings.Cut(commit, "\n")
		if slices.ContainsFunc(strings.Fields(files), func(file string) bool { return touchesSurface(file, closure) }) {
			moved++
		}
	}
	return last, moved, nil
}
