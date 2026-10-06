// Package landing lands one plan row through the small gate. The dirty tree
// is formatted, modernized by go fix and taken as the candidate, a Git tree
// written through a private index; the candidate is vetted and tested -- go
// test's own cache decides what runs again -- the lanes the change reaches
// run (in the host's lane turn, one landing at a time), a few rules compare
// the changed files with HEAD, the row's own check passes, and a change that
// reaches the benchmark runs the scorecard's inference slice; then that
// candidate, with the row's step removed from its plan, is committed onto
// the HEAD the landing found. Git holds the record: the message says what
// the gate measured, and the slice's trailer carries it for the next landing
// to compare against. A refusal names the stage that refused it; the loop
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
	// candidate is the tree the checks validate and the commit publishes.
	candidate string
	// planned is the plan the candidate holds and advanced is it with the
	// landed steps removed; verifiers and budget are the steps' checks and
	// the declared growth of their rows, summed over the distinct rows, as
	// that plan states them.
	planned   []byte
	advanced  plan.Plan
	verifiers []string
	budget    plan.Budget
}

// Run lands the row and returns the commit. Every check runs over the
// candidate; the first that fails refuses the landing and nothing is
// committed.
func Run(ctx context.Context, options Options) (string, error) {
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
	if options.Quiet {
		release, err := l.holdQuiet(ctx)
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
	if !slices.ContainsFunc(l.paths, func(changed string) bool { return changed != plan.Path }) {
		checks = slices.DeleteFunc(checks, func(check check) bool { return check.name != "verify" })
	}
	for _, check := range checks {
		started := time.Now()
		if err := check.run(ctx); err != nil {
			return "", fmt.Errorf("%s%s: %w", stagePrefix, check.name, err)
		}
		wall := time.Since(started).Round(time.Millisecond)
		entry := check.name + " " + wall.String()
		if detail := l.details[check.name]; detail != "" {
			entry += " (" + detail + ")"
		}
		l.measured = append(l.measured, entry)
		fmt.Fprintf(options.Log, "land: %s ok %s\n", check.name, wall)
	}
	if options.Preflight {
		return "", nil
	}
	return l.commit(ctx)
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

// shipSet takes the dirty tree as the candidate, then reads the paths it
// changes; a path that changes after the capture is one more path the
// checks read and the commit refuses. An untracked top-level entry HEAD does
// not hold is refused: a new root belongs in a change of its own.
func (l *landing) shipSet(ctx context.Context) error {
	var err error
	if l.candidate, err = l.capture(ctx); err != nil {
		return err
	}
	if err := l.row(ctx); err != nil {
		return err
	}
	entries, err := dirtyTree(ctx, l.Repo)
	if err != nil {
		return err
	}
	l.paths = nil
	tracked, err := gitauthority.Query(ctx, l.Repo, "ls-tree", "--name-only", "HEAD")
	if err != nil {
		return err
	}
	tops := strings.Split(strings.TrimSpace(string(tracked)), "\n")
	for _, entry := range entries {
		top, _, _ := strings.Cut(entry.Path, "/")
		if entry.IndexStatus == "?" && !slices.Contains(tops, top) {
			return fmt.Errorf("land: the dirty tree adds the untracked top-level entry %q; track it in its own change or remove it", top)
		}
		l.paths = append(l.paths, entry.Path)
		if entry.OriginalPath != "" {
			l.paths = append(l.paths, entry.OriginalPath)
		}
	}
	l.entries = entries
	return nil
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

// row reads the landed row from the plan the candidate holds, so the check
// the gate runs, the growth the rules admit and the plan the commit
// publishes all come from the bytes the gate validates.
func (l *landing) row(ctx context.Context) error {
	data, err := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.candidate+":"+plan.Path)
	if err != nil {
		return err
	}
	document, err := plan.Parse(data)
	if err != nil {
		return err
	}
	named, err := rows(l.Ref)
	if err != nil {
		return err
	}
	l.advanced, l.verifiers, l.budget = document, nil, plan.Budget{}
	counted := map[string]bool{}
	for _, row := range named {
		// A step's check and its row's declared growth are read from the plan before the step is removed.
		for _, item := range l.advanced.Items {
			if item.ID != row.item {
				continue
			}
			if !counted[item.ID] {
				counted[item.ID] = true
				l.budget.Nodes += item.Budget.Nodes
				l.budget.Reason = cmp.Or(l.budget.Reason, item.Budget.Reason)
			}
			for _, step := range item.Steps {
				if step.ID == row.step && !slices.Contains(l.verifiers, step.Verify) {
					l.verifiers = append(l.verifiers, step.Verify)
				}
			}
		}
		if l.advanced, err = plan.Advance(l.advanced, row.item, row.step); err != nil {
			return fmt.Errorf("land: %s is not an open step of the plan: %w", row, err)
		}
	}
	l.planned = data
	return nil
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
	if _, err := l.git(ctx, "update-ref", "-m", "land "+l.Ref, "HEAD", landed, l.head); err != nil {
		return "", fmt.Errorf("land: publish %s onto %.12s: %w", l.Ref, l.head, err)
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
	return landed, errors.Join(err, synced)
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
