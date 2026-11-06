package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/gosource"
	"overgo/internal/plan"
	"overgo/internal/scorecard"
)

// A cycle measures what the commands its cells run execute, so a commit moves what it measures
// only when it changes a Go package in the closure of one of them, a file such a package embeds, or a file the kernels live in. The closure is what go list reports, not a list kept by hand.
var (
	// surfaceCommands are the commands a cycle's cells run.
	surfaceCommands = []string{"./cmd/evaluate", "./cmd/train", "./cmd/longform", "./cmd/recipe", "./cmd/server", "./cmd/validate"}
	// surfacePrefixes are the paths outside Go packages a cell's computation reads.
	surfacePrefixes = []string{"kernels/"}
)

// surfaceClosure lists cycle packages and resolved embedded files through the package graph.
//   - declared patterns retain conservative coverage for deleted inputs
func surfaceClosure(ctx context.Context, root string) (map[string]bool, error) {
	graph, err := gosource.ListGraph(ctx, root, true, surfaceCommands...)
	if err != nil {
		return nil, fmt.Errorf("go list the cycle's commands: %w", err)
	}
	closure := map[string]bool{}
	for _, pkg := range graph {
		if inside, found := strings.CutPrefix(pkg.ImportPath, "overgo/"); found {
			closure[inside] = true
			for _, file := range pkg.EmbedFiles {
				closure[path.Join(inside, file)] = true
			}
			for _, pattern := range pkg.EmbedPatterns {
				closure["embed:"+path.Join(inside, strings.TrimPrefix(pattern, "all:"))] = true
			}
		}
	}
	return closure, nil
}

// touchesSurface reports whether a changed file moves what a cycle measures: a non-test Go file of a
// package in the closure, a file the closure names as embedded, or a path under a surface prefix.
func touchesSurface(root, file string, closure map[string]bool) bool {
	switch {
	case closure[file], slices.ContainsFunc(surfacePrefixes, func(prefix string) bool { return strings.HasPrefix(file, prefix) }):
		return true
	case strings.HasSuffix(file, ".go") && !strings.HasSuffix(file, "_test.go") && !strings.Contains(file, "/testdata/") && closure[path.Dir(file)]:
		return true
	}
	// Existing files use the resolved list. Missing files may have matched a
	// directory or glob before deletion, so retain that coverage conservatively.
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(file))); !errors.Is(err, os.ErrNotExist) {
		return err != nil
	}
	for key := range closure {
		pattern, embedded := strings.CutPrefix(key, "embed:")
		if !embedded {
			continue
		}
		for candidate := file; candidate != "."; candidate = path.Dir(candidate) {
			if matched, _ := path.Match(pattern, candidate); matched {
				return true
			}
		}
	}
	return false
}

// commitAbbreviation is how many hex digits of a commit a decision's reason shows.
const commitAbbreviation = 12

// unmovedSince is why no cycle is due while nothing a cycle measures has changed since the cycle at last, which
// completed, of any outcome: a failing cycle is not started again until what it measures changes.
func unmovedSince(last string) string {
	return "nothing a cycle measures has changed since the cycle at " + last[:min(len(last), commitAbbreviation)]
}

// HandStartDue says whether a cycle started by hand (-improve) is due, by the trigger's terms with the dispatched step
// beside them: a cycle a run left unfinished continues at its own commit, and otherwise one is due only when no cycle
// has completed, or what a cycle measures moved since the last that did and the plan is empty. While a step is
// dispatched none is: a fix is judged by its own check, and a decline by
// its own model's cells, never by a cycle of every model started for it (owner, 2026-10-31: stop retesting the whole
// scorecard without finishing). The second run of 2026-10-31 was started by hand after a fix to three records whose
// checks had passed on the card, and spent 9 hours 5 minutes.
func (w *improvement) HandStartDue(dispatch plan.Dispatch) (bool, string, error) {
	if begun := w.unfinished(); begun != "" {
		return true, "the cycle a run began at " + begun[:min(len(begun), commitAbbreviation)] + " continues", nil
	}
	last, moved, err := w.movedSince()
	if err != nil {
		return false, "", err
	}
	switch {
	case last == "":
		return true, "no cycle has completed, so there is no run to compare with", nil
	case moved == 0:
		return false, unmovedSince(last), nil
	case dispatch.Complete:
		return true, "the plan is empty and what a cycle measures moved", nil
	}
	return false, "the plan holds " + dispatch.Item + "/" + dispatch.Step + ", judged by its own check or its own model's cells, not by a run of every model", nil
}

// landingsThatMovedTheSurface counts the commits after the newest published scorecard's that touch
// what a cycle measures, and reports that scorecard's commit.
func landingsThatMovedTheSurface(ctx context.Context, root, records string, closure map[string]bool) (last string, moved int, err error) {
	published, err := scorecard.Records(records)
	if err != nil || len(published) == 0 {
		return "", 0, err
	}
	last = published[len(published)-1].Commit
	log, err := gitauthority.Query(ctx, root, "log", "--name-only", "-z", "--format=%x00%H", last+"..HEAD")
	if err != nil {
		return last, 0, err
	}
	for commit := range strings.SplitSeq(strings.TrimPrefix(string(log), "\x00"), "\x00\x00") {
		_, files, _ := strings.Cut(commit, "\x00")
		if slices.ContainsFunc(strings.Split(strings.TrimPrefix(files, "\n"), "\x00"), func(file string) bool { return file != "" && touchesSurface(root, file, closure) }) {
			moved++
		}
	}
	return last, moved, nil
}
