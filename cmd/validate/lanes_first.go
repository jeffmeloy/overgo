package main

import (
	"context"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
)

// lanesFirst is what a run of a lane takes first, read by the tool from the commits it runs (lanes-first): the tests
// the checks of the rows those commits landed name, newest commit first, each with the packages its check runs, and
// the packages the commits' changed files are in, whose checkpoint tests follow. moss-attention landed at 00:46 on
// 2026-11-19 and the checkpoints of the model it changed failed at 13:57, two hours into a run that reached them in
// its own order.
type lanesFirst struct {
	checks   []checkedTest
	packages map[string]bool
}

// checkedTest is a test a landed row's check names, and the packages its check runs.
type checkedTest struct {
	name     string
	packages []string
}

// readLanesFirst reads what a run of a lane at head takes first from the commits after base: each commit's
// Landed-Row trailers, each step's check as the plan held it before the commit landed it, and the files changed.
func readLanesFirst(ctx context.Context, root, base, head string) (lanesFirst, error) {
	first := lanesFirst{packages: map[string]bool{}}
	listed, err := gitauthority.Query(ctx, root, "log", "--format=%H", base+".."+head)
	if err != nil {
		return first, err
	}
	for commit := range strings.FieldsSeq(string(listed)) {
		body, err := gitauthority.Query(ctx, root, "show", "-s", "--format=%B", commit)
		if err != nil {
			return first, err
		}
		var landed []string
		for line := range strings.Lines(string(body)) {
			if row, found := strings.CutPrefix(strings.TrimSpace(line), "Landed-Row: "); found {
				landed = append(landed, row)
			}
		}
		if len(landed) == 0 {
			continue
		}
		published, err := gitauthority.Query(ctx, root, "show", commit+"^:"+plan.Path)
		if err != nil {
			return first, err
		}
		document, err := plan.Parse(published)
		if err != nil {
			return first, err
		}
		for _, row := range landed {
			item, step, _ := strings.Cut(row, "/")
			for _, open := range document.Items {
				for _, held := range open.Steps {
					if open.ID == item && held.ID == step {
						first.checks = append(first.checks, testsOfCheck(held.Verify)...)
					}
				}
			}
		}
	}
	changed, err := gitauthority.Query(ctx, root, "diff", "--name-only", base, head)
	if err != nil {
		return first, err
	}
	for file := range strings.FieldsSeq(string(changed)) {
		first.packages["./"+path.Dir(file)] = true
	}
	return first, nil
}

// testsOfCheck are the tests a go test check names with -run, a pattern ^(A|B)$ or ^A$, each with the packages the
// check runs; a check naming none gives none.
func testsOfCheck(check string) []checkedTest {
	words := strings.Fields(check)
	var packages, names []string
	for index, word := range words {
		switch {
		case strings.HasPrefix(word, "./"):
			packages = append(packages, word)
		case word == "-run" && index+1 < len(words):
			pattern := strings.Trim(words[index+1], `'"`)
			pattern = strings.TrimSuffix(strings.TrimPrefix(pattern, "^"), "$")
			pattern = strings.TrimSuffix(strings.TrimPrefix(pattern, "("), ")")
			names = strings.Split(pattern, "|")
		}
	}
	var tests []checkedTest
	for _, name := range names {
		if strings.HasPrefix(name, "Test") && !strings.ContainsAny(name, `.*+?[]{}\`) {
			tests = append(tests, checkedTest{name: name, packages: packages})
		}
	}
	return tests
}

// rank is where a run takes a package's test: the place of the first landed check naming it in the package, then the
// checkpoint tests of a changed package, then the rest, each after those before it.
func (f lanesFirst) rank(pkg, name string) int {
	for index, check := range f.checks {
		if check.name == name && slices.Contains(check.packages, pkg) {
			return index
		}
	}
	if f.packages[pkg] && strings.Contains(name, "Checkpoint") {
		return len(f.checks)
	}
	return len(f.checks) + 1
}

// yieldTo is the Yields of a run a runner starts (lanes-first/yield): the commit the due file at due names, where it
// is another than the run's own, which the runner names due as a newer close lands; the run's record of yielding is
// written beside the file first (plan.LanesYielded), and a run that cannot write it does not yield.
func yieldTo(ctx context.Context, root, due string) func() (string, bool) {
	head, headErr := gitauthority.Query(ctx, root, "rev-parse", "HEAD")
	own := strings.TrimSpace(string(head))
	return func() (string, bool) {
		var named struct {
			Commit string `json:"commit"`
		}
		if headErr != nil || own == "" || jsonfile.Decode(due, &named) != nil || named.Commit == "" || named.Commit == own {
			return "", false
		}
		record := plan.LanesYielded{Commit: own, To: named.Commit}
		if jsonfile.Write(filepath.Join(filepath.Dir(due), plan.LanesYieldedFile), record, clioptions.PrivateFileMode) != nil {
			return "", false
		}
		return named.Commit, true
	}
}

// lanesYield is the Yields of the run a runner started, which names its due file in plan.LanesDueEnvironment
// (yieldTo); a run no runner started runs to its end.
func lanesYield(ctx context.Context, root string) func() (string, bool) {
	due := os.Getenv(plan.LanesDueEnvironment)
	if due == "" {
		return nil
	}
	return yieldTo(ctx, root, due)
}
