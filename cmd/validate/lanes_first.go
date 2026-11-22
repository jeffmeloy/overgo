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

// checkedTest is a test a landed row's check names, the packages its check runs, and the commit that landed the row.
type checkedTest struct {
	name     string
	packages []string
	commit   string
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
						for _, check := range testsOfCheck(held.Verify) {
							check.commit = commit
							first.checks = append(first.checks, check)
						}
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

// passedSince leaves out of the first list each test with a pass kept (plan.ReadTestPasses, by plan.PassedTest) in
// every package its check runs at the commit that landed the row naming it or at one after it (since: the later
// commit is the earlier or descends from it), and says how many it left out: such a test runs in its package's own
// order. The first list is read from the commits after the lane's last whole pass, so a lane that has not passed whole
// repeated the same tests first at every close: the models lane, unpassed whole since 2026-11-16, took 89 tests first
// at each run, two hours and more, though the run before had kept passes of them, and commits were held for it to reach
// internal/inference (lanes-first-kept, the owner's review of 2026-11-21). A test whose check names no package keeps
// its place.
func (f *lanesFirst) passedSince(passes map[string]plan.TestPass, since func(earlier, later string) bool) int {
	kept := f.checks[:0]
	for _, check := range f.checks {
		passed := len(check.packages) > 0 && check.commit != ""
		for _, pkg := range check.packages {
			pass, held := passes[plan.PassedTest(modulePrefix+strings.TrimPrefix(pkg, "./"), check.name)]
			passed = passed && held && since(check.commit, pass.Commit)
		}
		if !passed {
			kept = append(kept, check)
		}
	}
	left := len(f.checks) - len(kept)
	f.checks = kept
	return left
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

// runnerYield is the yield of a run a runner starts to the commit its due file names (lanes-first/yield): the run's own
// commit, the due file, and the reading of the run's own commit.
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

// yielded writes the run's record of yielding to to beside the due file, with the lanes it failed before
// (plan.LanesYielded), which the runner reads to say how the run ended.
func (y runnerYield) yielded(to string, failures []string) error {
	record := plan.LanesYielded{Commit: y.own, To: to, Failures: failures}
	return jsonfile.Write(filepath.Join(filepath.Dir(y.due), plan.LanesYieldedFile), record, clioptions.PrivateFileMode)
}

// lanesYield is the yield of the run a runner started, which names its due file in plan.LanesDueEnvironment
// (yieldTo); a run no runner started runs to its end, given neither.
func lanesYield(ctx context.Context, root string) (func() (string, bool), func(string, []string) error) {
	due := os.Getenv(plan.LanesDueEnvironment)
	if due == "" {
		return nil, nil
	}
	yield := yieldTo(ctx, root, due)
	return yield.yields, yield.yielded
}
