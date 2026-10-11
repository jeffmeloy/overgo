// Package landing lands one plan row through the small gate. The dirty tree
// is formatted, modernized by go fix and taken as the candidate, a Git tree
// written through a private index; the candidate is vetted and tested -- go
// test's own cache decides what runs again -- the lanes the change reaches
// run (in the host's lane turn, one landing at a time), a few rules compare
// the changed files with HEAD, the row's own check passes, and a change that
// reaches the benchmark runs the scorecard's inference slice; then that
// candidate, with the row's step removed from its plan, is committed onto
// the HEAD the landing found. The check and the count of production code the
// row may add are the ones the plan at that HEAD gives it, not the ones the
// change carries, and the plan the commit publishes holds what the landing
// left of the count. Git holds the record: the message says what
// the gate measured, and the slice's trailer carries it for the next landing
// to compare against. A run that may commit holds the repository's one turn
// from its start to its end, so the commits begun in a repository's worktrees
// run one at a time (holdTurn). A worktree that is not the trunk's own
// checkout is first brought up to the trunk's tip, its change merged onto
// what landed since it was written (integrate); its commit is then checked on
// that tree by the tip's terms and published onto the trunk, whose checkout
// follows. A refusal names the stage that refused it; the loop
// classes it, an antivirus quarantine as an external prerequisite, and
// records its class in the checkout's process state, where a recurrence shows.
package landing

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/processcontrol"
	"overgo/internal/repoanalysis"
	"overgo/internal/testevidence"
)

// Options names the row a landing serves, its authored message, and where
// progress goes.
type Options struct {
	Repo    string
	Ref     string
	Message []byte
	Log     io.Writer
	// Preflight runs only the stages that are cheap, over the same candidate
	// and with the same refusals, and commits nothing: what the long stages
	// would refuse for a reason the cheap ones know is found in seconds.
	Preflight bool
	// Quiet holds the host's quiet machine for the landing's whole run, waiting for the
	// timed measurement in progress. The landing command sets it; a fixture does not, so
	// a test of the gate never waits for a cycle's cell.
	Quiet bool
}

type landing struct {
	Options
	// head is the commit the landing found and publishes onto; merging are
	// the heads a merge in progress joins to it.
	head    string
	merging []string
	// scratch holds the private index the candidate is written through.
	scratch  string
	entries  []repoanalysis.DirtyPath
	paths    []string
	measured []string
	// details say what a stage measured beyond its time: the packages the
	// test stage ran rather than replayed, what the scorecard slice found.
	details map[string]string
	// trailers are the scorecard slices' commit trailers, one per slice
	// that measured or failed.
	trailers []string
	// trunk is the directory of the repository's main worktree and branch the
	// branch it has checked out; onto is that branch when this checkout is
	// not the one that has it, and the commit publishes onto it.
	trunk, branch, onto string
	// candidate is the tree the checks validate and the commit publishes.
	candidate string
	// planned is the plan the candidate holds and advanced is it with the
	// landed steps removed. terms is the plan the change is judged by, and
	// verifiers the landed steps' checks as it states them.
	planned   []byte
	advanced  plan.Plan
	terms     plan.Plan
	verifiers []string
	// spent is where the run's time went, for the checkout's record.
	spent spent
}

// Run lands the row and returns the commit. Every check runs over the
// candidate; the first that fails refuses the landing and nothing is
// committed.
func Run(ctx context.Context, options Options) (landed string, err error) {
	if _, err := rows(options.Ref); err != nil {
		return "", err
	}
	root, err := filepath.Abs(options.Repo)
	if err != nil {
		return "", err
	}
	options.Repo = root
	if err := requireSections(options.Message); err != nil {
		return "", err
	}
	scratch, err := os.MkdirTemp("", "overgo-landing-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(scratch)
	l := &landing{Options: options, scratch: scratch, details: map[string]string{}}
	// Where the run's time went is recorded however it ends. A record that
	// cannot be written is said and changes nothing: the landing stands.
	l.spent = spent{Reference: options.Ref, Started: time.Now(), Outcome: gateClass, Stages: map[string]time.Duration{}}
	defer func() {
		if unrecorded := l.recordSpent(); unrecorded != nil {
			fmt.Fprintf(options.Log, "land: where the run's time went was not recorded: %v\n", unrecorded)
		}
	}()
	// A run that may commit takes the repository's one turn first, and holds it to its end.
	if !options.Preflight {
		release, err := l.holdTurn(ctx)
		if err != nil {
			return "", err
		}
		defer func() { _ = release() }()
		// Under the turn the trunk stands still: a worktree that is behind it is brought up to it.
		if err := l.integrate(ctx); err != nil {
			return "", err
		}
	}
	if options.Quiet {
		release, err := l.holdQuiet(ctx, quietMachine)
		if err != nil {
			return "", err
		}
		defer func() { _ = release() }()
	}
	if err := l.parents(ctx); err != nil {
		return "", err
	}
	if err := l.shipSet(ctx); err != nil {
		return "", err
	}
	checks := l.checks()
	if options.Preflight {
		checks = slices.DeleteFunc(checks, func(check check) bool { return !check.cheap })
	}
	// A change to the plan alone is data the row read above has validated;
	// nothing compiles it, so only the row's verify can speak to it, and
	// the tests that read the plan run with the next code landing.
	if l.planOnly() {
		checks = slices.DeleteFunc(checks, func(check check) bool { return check.name != "verify" })
	}
	for _, check := range checks {
		started := time.Now()
		l.spent.Outcome = check.name
		err := check.run(ctx)
		wall := time.Since(started).Round(time.Millisecond)
		if l.spent.Stages[check.name] = wall; err != nil {
			return "", fmt.Errorf("%s%s: %w", stagePrefix, check.name, err)
		}
		entry := check.name + " " + wall.String()
		if detail := l.details[check.name]; detail != "" {
			entry += " (" + detail + ")"
		}
		l.measured = append(l.measured, entry)
		// What a stage measured is said as it passes, so a preflight shows it too.
		fmt.Fprintf(options.Log, "land: %s ok %s\n", check.name, strings.TrimPrefix(entry, check.name+" "))
	}
	if options.Preflight {
		l.spent.Outcome = outcomePreflight
		return "", nil
	}
	// Every stage passed: a refusal from here is the commit's.
	l.spent.Outcome = outcomeConflict
	if landed, err = l.commit(ctx); landed != "" {
		l.spent.Outcome = outcomeLanded
	}
	return landed, err
}

// check is one stage of the gate; cheap says the preflight runs it.
type check struct {
	name  string
	run   func(context.Context) error
	cheap bool
}

// checks are the gate's stages in the order they run; a refusal names the
// stage that refused, and Classify reads the name back. The cheap ones are
// the stages that take seconds and refuse what the change itself got wrong:
// the files' form, the style and growth rules, and the invariant tests the
// repository holds itself to, which also run first in a full landing so the
// refusal comes before the tests of every package.
func (l *landing) checks() []check {
	return []check{
		{"format", l.format, true}, {"fix", l.fix, true}, {"vet", l.vet, true}, {"rules", l.rules, true},
		{"invariants", l.invariants, true}, {"test", l.test, false}, {"lanes", l.lanes, false},
		{"verify", l.verify, true}, {"scorecard", l.scorecard, false},
	}
}

// stagePrefix opens the refusal of a stage, before the stage's name.
const stagePrefix = "land: "

// predictedSection opens the part of a landing message that says what the
// change should do.
const predictedSection = "Predicted effect:"

// requireSections holds the message to the two sections every landing
// states: why the change was needed and what it should do.
func requireSections(message []byte) error {
	for _, label := range []string{"Cause:", predictedSection} {
		if _, found := section(message, label); !found {
			return fmt.Errorf("land: the message lacks a %q section", label)
		}
	}
	return nil
}

// section is the message from the line that opens with label to its end,
// and whether a line does.
func section(message []byte, label string) (string, bool) {
	lines := strings.Split(string(message), "\n")
	at := slices.IndexFunc(lines, func(line string) bool { return strings.HasPrefix(strings.TrimSpace(line), label) })
	if at < 0 {
		return "", false
	}
	return strings.Join(lines[at:], "\n"), true
}

// parents reads the commit the landing publishes onto, seeding the private
// index from it, and the heads a merge in progress joins to it: a merge
// lands as the merge.
func (l *landing) parents(ctx context.Context) error {
	head, err := gitauthority.Query(ctx, l.Repo, "rev-parse", "HEAD")
	if err != nil {
		return err
	}
	l.head = strings.TrimSpace(string(head))
	if _, err := l.staged(ctx, "read-tree", l.head); err != nil {
		return err
	}
	merge, err := gitauthority.Query(ctx, l.Repo, "rev-parse", "--path-format=absolute", "--git-path", "MERGE_HEAD")
	if err != nil {
		return err
	}
	merging, err := os.ReadFile(strings.TrimSpace(string(merge)))
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	l.merging = strings.Fields(string(merging))
	return nil
}

// builtOn names the commit a change is read against: the one the landing
// found and publishes onto, which a worktree that was brought up to the trunk
// has as its HEAD, or HEAD itself for a stage run before it is read.
func (l *landing) builtOn() string { return cmp.Or(l.head, "HEAD") }

// shipSet takes the dirty tree as the candidate, then reads the paths it
// changes; a path that changes after the capture is one more path the
// checks read and the commit refuses. An untracked top-level entry HEAD does
// not hold is refused: a new root belongs in a change of its own.
func (l *landing) shipSet(ctx context.Context) error {
	var err error
	if l.candidate, err = l.capture(ctx); err != nil {
		return err
	}
	entries, err := dirtyTree(ctx, l.Repo)
	if err != nil {
		return err
	}
	l.paths = nil
	for _, entry := range entries {
		l.paths = append(l.paths, entry.Path)
		if entry.OriginalPath != "" {
			l.paths = append(l.paths, entry.OriginalPath)
		}
	}
	if err := l.row(ctx); err != nil {
		return err
	}
	tracked, err := gitauthority.Query(ctx, l.Repo, "ls-tree", "--name-only", l.builtOn())
	if err != nil {
		return err
	}
	tops := strings.Split(strings.TrimSpace(string(tracked)), "\n")
	for _, entry := range entries {
		top, _, _ := strings.Cut(entry.Path, "/")
		if entry.IndexStatus == "?" && !slices.Contains(tops, top) {
			return fmt.Errorf("land: the dirty tree adds the untracked top-level entry %q; track it in its own change or remove it", top)
		}
	}
	l.entries = entries
	return nil
}

// planOnly reports a change to the plan alone.
func (l *landing) planOnly() bool {
	return !slices.ContainsFunc(l.paths, func(changed string) bool { return changed != plan.Path })
}

// capture writes the working tree, as git add -A stages it, through the
// landing's private index and returns its tree. The index the checkout's own
// git commands use is neither read nor written.
func (l *landing) capture(ctx context.Context) (string, error) {
	if _, err := l.staged(ctx, "add", "-A"); err != nil {
		return "", err
	}
	tree, err := l.staged(ctx, "write-tree")
	return strings.TrimSpace(tree), err
}

// row reads the plan the candidate holds, which the commit publishes with the
// landed steps removed, and the terms the change is judged by: each landed
// step's check and what its row may still add. The terms come from the plan
// of the commit the landing builds on, so a change that edits its own row's
// count or check is held to the ones it started with, and a step the change
// itself adds has no terms to land by. Two changes have no earlier terms and
// are judged by the plan they hold: a change to the plan alone, which is how
// a row is filed and a count raised, and a merge, whose row is named for the
// head it joins.
func (l *landing) row(ctx context.Context) error {
	data, err := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.candidate+":"+plan.Path)
	if err != nil {
		return err
	}
	document, err := plan.Parse(data)
	if err != nil {
		return err
	}
	l.terms = document
	if len(l.merging) == 0 && !l.planOnly() {
		held, err := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.head+":"+plan.Path)
		if err != nil {
			return err
		}
		if l.terms, err = plan.Parse(held); err != nil {
			return err
		}
	}
	named, err := rows(l.Ref)
	if err != nil {
		return err
	}
	l.advanced, l.verifiers = document, nil
	for _, row := range named {
		judged := false
		for _, item := range l.terms.Items {
			for _, step := range item.Steps {
				if item.ID != row.item || step.ID != row.step {
					continue
				}
				if judged = true; !slices.Contains(l.verifiers, step.Verify) {
					l.verifiers = append(l.verifiers, step.Verify)
				}
			}
		}
		if !judged {
			return fmt.Errorf("land: %s is not a step of the plan at %.12s, the commit this change builds on, and a change is judged by the count and the check its row had before it; land the plan's edit alone first", row, l.head)
		}
		if l.advanced, err = plan.Advance(l.advanced, row.item, row.step); err != nil {
			return fmt.Errorf("land: %s is not an open step of the plan: %w", row, err)
		}
	}
	l.planned = data
	return nil
}

// charge takes the change's growth from the count of the row it lands, in the
// plan the commit publishes, and returns what the landed rows could add
// before it. A change that shrinks production code adds what it removed to
// the count, so a row may delete in one landing and add in the next what one
// landing could do at once. Rows landed together are admitted by the sum of
// their counts and charged as one: the growth is taken from the first of them
// the plan still holds and the others keep their own, so rows whose growth is
// to be told apart land apart.
func (l *landing) charge(growth int) (left int) {
	named, _ := rows(l.Ref)
	for _, item := range l.terms.Items {
		if !slices.ContainsFunc(named, func(row ref) bool { return row.item == item.ID }) {
			continue
		}
		left += item.Budget.Nodes
		if at := slices.IndexFunc(l.advanced.Items, func(open plan.Item) bool { return open.ID == item.ID }); at >= 0 {
			l.advanced.Items[at].Budget.Nodes, growth = item.Budget.Nodes-growth, 0
		}
	}
	return left
}

// ref names one plan step a landing completes.
type ref struct{ item, step string }

// String is the step as a landing names it, item/step.
func (r ref) String() string { return r.item + "/" + r.step }

// rows reads the steps a landing names: item/step, or several of them joined by commas, which
// land together in one commit, each with its trailer, under one gate run. A step named twice, or
// a name that is not item/step, is refused.
func rows(reference string) ([]ref, error) {
	var named []ref
	for part := range strings.SplitSeq(reference, ",") {
		item, step, found := strings.Cut(strings.TrimSpace(part), "/")
		if !found || item == "" || step == "" || strings.Contains(step, "/") {
			return nil, fmt.Errorf("land: %q is not item/step, or item/step steps joined by commas", reference)
		}
		next := ref{item, step}
		if slices.Contains(named, next) {
			return nil, fmt.Errorf("land: %s is named twice in %q", next, reference)
		}
		named = append(named, next)
	}
	return named, nil
}

func dirtyTree(ctx context.Context, repo string) ([]repoanalysis.DirtyPath, error) {
	raw, err := gitauthority.Query(ctx, repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return nil, err
	}
	return repoanalysis.ParseDirtyStatus(raw)
}

// goPackages are the directories of changed Go files that still exist.
func (l *landing) goPackages() []string {
	var directories []string
	for _, name := range l.paths {
		if !strings.HasSuffix(name, ".go") || slices.Contains(strings.Split(name, "/"), "testdata") {
			continue
		}
		if _, err := os.Stat(filepath.Join(l.Repo, filepath.FromSlash(name))); err != nil {
			continue
		}
		directories = append(directories, "./"+path.Dir(name))
	}
	slices.Sort(directories)
	return slices.Compact(directories)
}

func (l *landing) format(ctx context.Context) error {
	packages := l.goPackages()
	if len(packages) == 0 {
		return nil
	}
	_, err := l.run(ctx, nil, "go", append([]string{"fmt"}, packages...)...)
	return err
}

// fix applies go fix's modernizers to the changed packages, as format
// applies gofmt: a legacy form go fix can rewrite is rewritten, not refused.
// A directory whose files this platform's build excludes, or that does not
// compile, has no package to fix; vet reports the latter. Both rewrite
// whole packages, so a file they rewrote that the change left untouched
// joins the ship set here rather than reading as a tree that moved.
func (l *landing) fix(ctx context.Context) error {
	directories := l.goPackages()
	if len(directories) != 0 {
		listed, err := l.run(ctx, nil, "go", append([]string{"list", "-e", "-tags", CycleTags, "-f", "{{if not .Error}}{{.ImportPath}}{{end}}"}, directories...)...)
		if err != nil {
			return err
		}
		if packages := strings.Fields(listed); len(packages) != 0 {
			if _, err := l.run(ctx, nil, "go", append([]string{"fix", "-tags", CycleTags}, packages...)...); err != nil {
				return err
			}
		}
		// What the source derives is regenerated, as go fix is applied, not
		// refused for a landing again after.
		for _, generator := range l.generators() {
			if _, err := l.run(ctx, generator.environment, "go", generator.arguments...); err != nil {
				return err
			}
		}
	}
	return l.shipSet(ctx)
}

// generator is a command that rewrites a file the source derives.
type generator struct {
	arguments, environment []string
}

// generators are the derived-file commands the repository has, in the order
// they run: the API manifest, regenerated from the source, and the
// compatibility claims, each rebound only by its own verify passing, which
// takes a moment when no bound source moved. A repository without the
// command has nothing to derive with it.
func (l *landing) generators() []generator {
	var held []generator
	for _, candidate := range []struct {
		command string
		generator
	}{
		{"api-manifest", generator{arguments: []string{"run", "./cmd/api-manifest", "-update"}}},
		{"validate", generator{arguments: []string{"run", "./cmd/validate", "-claims", "-run"}, environment: append(os.Environ(), cudaTestEnvironment)}},
	} {
		if _, err := os.Stat(filepath.Join(l.Repo, "cmd", candidate.command)); err == nil {
			held = append(held, candidate.generator)
		}
	}
	return held
}

func (l *landing) vet(ctx context.Context) error {
	_, err := l.run(ctx, nil, "go", "vet", "-tags", CycleTags, "./...")
	return err
}

// test runs every package; go test's cache replays a package whose inputs
// did not change. No timeout is imposed: a test that must bound its own
// wait does so.
func (l *landing) test(ctx context.Context) error {
	return l.goTest(ctx, "test", "./...")
}

// invariantPackages are the packages whose tests hold the repository to its
// own rules: what may be written where, which commands and documents refer to
// what, and how a landing and the plan are shaped. A change that breaks one is
// refused here, ahead of every other package's tests, and by the preflight.
var invariantPackages = []string{"./internal/repoanalysis", "./internal/commanddoc", "./internal/landing", "./internal/plan"}

// invariants runs the invariant packages' tests alone, those the repository
// has: a tree without them, a fixture's, has no invariant to hold.
func (l *landing) invariants(ctx context.Context) error {
	held := slices.DeleteFunc(slices.Clone(invariantPackages), func(pattern string) bool {
		_, err := os.Stat(filepath.Join(l.Repo, filepath.FromSlash(pattern)))
		return err != nil
	})
	if len(held) == 0 {
		return nil
	}
	return l.goTest(ctx, "invariants", held...)
}

// goTest runs the tests of packages and says which of them ran rather than
// replayed, under the stage's name.
func (l *landing) goTest(ctx context.Context, stage string, packages ...string) error {
	output, err := l.run(ctx, nil, "go", append([]string{"test", "-json", "-timeout=0"}, packages...)...)
	if err != nil {
		if summary := testevidence.FailureSummary(output); summary != "" {
			return fmt.Errorf("failed tests/packages: %s", summary)
		}
		return err
	}
	l.details[stage] = ranPackages(output)
	return nil
}

// reportedPackages is how many of the packages go test ran the measurement
// names: the slowest few are where a landing's test time went.
const reportedPackages = 5

// cachedMarker ends go test's line for a package it replayed from its cache
// rather than ran.
const cachedMarker = "\t(cached)"

// ranPackages names the packages go test ran rather than replayed from its
// cache, slowest first, so the measured test time says what it was spent on.
func ranPackages(output string) string {
	type event struct {
		Action, Package, Test, Output string
		Elapsed                       float64
	}
	cached := map[string]bool{}
	var ran []event
	for line := range strings.Lines(output) {
		var parsed event
		if json.Unmarshal([]byte(line), &parsed) != nil || parsed.Test != "" {
			continue
		}
		switch {
		case parsed.Action == "output" && strings.Contains(parsed.Output, cachedMarker):
			cached[parsed.Package] = true
		case (parsed.Action == "pass" || parsed.Action == "fail") && !cached[parsed.Package]:
			ran = append(ran, parsed)
		}
	}
	slices.SortFunc(ran, func(left, right event) int { return cmp.Compare(right.Elapsed, left.Elapsed) })
	var slowest []string
	for _, run := range ran[:min(len(ran), reportedPackages)] {
		slowest = append(slowest, fmt.Sprintf("%s %.0fs", strings.TrimPrefix(run.Package, "overgo/"), run.Elapsed))
	}
	return fmt.Sprintf("ran %d packages: %s", len(ran), strings.Join(slowest, ", "))
}

func (l *landing) verify(ctx context.Context) error {
	for _, verifier := range l.verifiers {
		if strings.TrimSpace(verifier) == "" {
			continue
		}
		if _, err := planverify.Execute(ctx, l.Repo, verifier, nil); err != nil {
			return err
		}
	}
	return nil
}

// regularFile is the Git mode the advanced plan is staged with.
const regularFile = "100644"

// commit publishes the candidate the checks passed, its plan advanced past
// the row, onto the HEAD the landing found. The working tree must still be
// the candidate, and HEAD moves by one compare-and-set, so a refusal writes
// nothing: the plan keeps its row and any edit made meanwhile. Only a
// landed commit advances the working plan, and only while that plan is
// still the candidate's; a plan edited during the landing is left as
// edited and the landing says so.
func (l *landing) commit(ctx context.Context) (landed string, err error) {
	lock, err := authoritylock.AcquireContext(ctx, l.Repo)
	if err != nil {
		return "", err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	now, err := l.capture(ctx)
	if err != nil {
		return "", err
	}
	if now != l.candidate {
		changed, err := gitauthority.Query(ctx, l.Repo, "diff-tree", "-r", "--name-only", l.candidate, now)
		if err != nil {
			return "", err
		}
		return "", fmt.Errorf("land: the tree changed during the landing: %s; land again once it is settled",
			strings.ReplaceAll(strings.TrimSpace(string(changed)), "\n", ", "))
	}
	advanced := filepath.Join(l.scratch, "plan.json")
	if err := plan.Save(advanced, l.advanced); err != nil {
		return "", err
	}
	blob, err := l.staged(ctx, "hash-object", "-w", "--path="+plan.Path, advanced)
	if err != nil {
		return "", err
	}
	if _, err := l.staged(ctx, "update-index", "--cacheinfo", regularFile+","+strings.TrimSpace(blob)+","+plan.Path); err != nil {
		return "", err
	}
	tree, err := l.staged(ctx, "write-tree")
	if err != nil {
		return "", err
	}
	message := filepath.Join(l.scratch, "message.txt")
	if err := os.WriteFile(message, []byte(l.message()), clioptions.PrivateFileMode); err != nil {
		return "", err
	}
	arguments := []string{"commit-tree", strings.TrimSpace(tree), "-F", message, "-p", l.head}
	for _, parent := range l.merging {
		arguments = append(arguments, "-p", parent)
	}
	commit, err := l.git(ctx, arguments...)
	if err != nil {
		return "", err
	}
	landed = strings.TrimSpace(commit)
	// From a worktree that is not the trunk's own checkout the trunk moves
	// first, and only if its checkout can follow: its index and files are read
	// forward from the old tip to the new, which uncommitted changes of the
	// lead's to a file this commit changes refuse.
	follow := func(trial ...string) error {
		_, err := l.git(ctx, slices.Concat([]string{"-C", l.trunk, "read-tree"}, trial, []string{"-m", "-u", l.head, landed})...)
		return err
	}
	if l.onto != "" {
		// A file written and put back is as git holds it, which only a refreshed index knows; a file that differs stays so, and the refresh says nothing of it here.
		_, _ = l.git(ctx, "-C", l.trunk, "update-index", "-q", "--refresh")
		if err := follow("-n"); err != nil {
			return "", fmt.Errorf("land: the trunk's checkout %s cannot follow %s, having uncommitted changes of its own to what it changes; they are committed or put aside there first: %w", l.trunk, l.Ref, err)
		}
		if _, err := l.git(ctx, "update-ref", "-m", "land "+l.Ref, l.onto, landed, l.head); err != nil {
			return "", fmt.Errorf("land: publish %s onto the trunk %s at %.12s: %w", l.Ref, l.onto, l.head, err)
		}
	}
	if _, err := l.git(ctx, "update-ref", "-m", "land "+l.Ref, "HEAD", landed, l.head); err != nil {
		return "", fmt.Errorf("land: publish %s onto %.12s: %w", l.Ref, l.head, err)
	}
	var followed error
	if l.onto != "" {
		if followed = follow(); followed != nil {
			followed = fmt.Errorf("land: %s landed as %.12s, but the trunk's checkout %s did not follow it and holds the files of %.12s: %w", l.Ref, landed, l.trunk, l.head, followed)
		}
	}
	planPath := filepath.Join(l.Repo, filepath.FromSlash(plan.Path))
	held, err := os.ReadFile(planPath)
	switch {
	case err != nil:
	case !bytes.Equal(held, l.planned):
		err = fmt.Errorf("land: %s landed as %.12s, but %s changed during the landing and is left as edited; reconcile it with HEAD's", l.Ref, landed, plan.Path)
	default:
		err = plan.Save(planPath, l.advanced)
	}
	// The checkout's index follows the commit, and a merge it concluded ends.
	_, synced := l.git(ctx, "reset", "-q")
	return landed, errors.Join(err, followed, synced)
}

// message is the authored message, what the gate measured, and the trailers:
// the scorecard slices, then the landed row.
func (l *landing) message() string {
	trailers := slices.Clone(l.trailers)
	named, _ := rows(l.Ref)
	for _, row := range named {
		trailers = append(trailers, plan.LandedRowTrailer+": "+row.String())
	}
	return strings.TrimSpace(string(l.Message)) + "\n\nMeasured by the gate: " + strings.Join(l.measured, ", ") +
		"\n\n" + strings.Join(trailers, "\n") + "\n"
}

func (l *landing) git(ctx context.Context, arguments ...string) (string, error) {
	return l.run(ctx, gitauthority.RepositoryEnvironment(), "git", gitauthority.WriterArguments(arguments...)...)
}

// staged runs git over the landing's private index.
func (l *landing) staged(ctx context.Context, arguments ...string) (string, error) {
	environment := append(gitauthority.RepositoryEnvironment(), "GIT_INDEX_FILE="+filepath.Join(l.scratch, "index"))
	return l.run(ctx, environment, "git", gitauthority.WriterArguments(arguments...)...)
}

// run executes one command at the repository root and returns its standard
// output; a failure carries the tail of both streams. Each stream has its
// own buffer: the two are drained concurrently.
func (l *landing) run(ctx context.Context, environment []string, name string, arguments ...string) (string, error) {
	// Every command says whose it is, so a measurement it runs does not wait for this landing.
	if environment == nil {
		environment = os.Environ()
	}
	environment = append(slices.Clone(environment), underLandingEnvironment+"=1")
	var output, diagnostics bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: name, Args: arguments, Dir: l.Repo, Env: environment, Stdout: &output, Stderr: &diagnostics,
	})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("%s %s: exit %d: %s", name, strings.Join(arguments, " "), receipt.ExitCode, tail(output.String()+diagnostics.String()))
	}
	return output.String(), err
}
