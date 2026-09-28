// Package landing lands one plan row through the small gate. The dirty tree
// is formatted, modernized by go fix, vetted and tested -- go test's own
// cache decides what runs again -- the lanes the change reaches run, a few
// rules compare the changed files with HEAD, the row's own check passes, and
// a change that reaches the benchmark runs the scorecard's slices, one per
// task, beside HEAD's benchmark when the message predicts a timed metric;
// then the tree is committed with the row's step removed from the plan. Git
// holds the record: the message says what the gate measured, and a trailer
// per slice carries it for the next landing to compare against.
package landing

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

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
}

type landing struct {
	Options
	item, step string
	head       string
	entries    []repoanalysis.DirtyPath
	paths      []string
	measured   []string
	// details say what a stage measured beyond its time: the packages the
	// test stage ran rather than replayed, what the scorecard slice found.
	details map[string]string
	// trailers are the scorecard slices' commit trailers, one per slice
	// that measured or failed.
	trailers []string
	// budget is the growth the landed row declared in the plan.
	budget plan.Budget
}

// Run lands the row and returns the commit. Every check runs over the dirty
// tree; the first that fails refuses the landing and nothing is committed.
func Run(ctx context.Context, options Options) (string, error) {
	item, step, found := strings.Cut(options.Ref, "/")
	if !found {
		return "", fmt.Errorf("land: %q is not item/step", options.Ref)
	}
	root, err := filepath.Abs(options.Repo)
	if err != nil {
		return "", err
	}
	options.Repo = root
	l := &landing{Options: options, item: item, step: step, details: map[string]string{}}
	if err := requireSections(options.Message); err != nil {
		return "", err
	}
	head, err := gitauthority.Query(ctx, options.Repo, "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	l.head = strings.TrimSpace(string(head))
	document, err := plan.Load(filepath.Join(options.Repo, plan.Path))
	if err != nil {
		return "", err
	}
	if _, err := plan.Advance(document, item, step); err != nil {
		return "", fmt.Errorf("land: %s is not an open step of the plan: %w", options.Ref, err)
	}
	verify := rowVerify(document, item, step)
	if at := slices.IndexFunc(document.Items, func(candidate plan.Item) bool { return candidate.ID == item }); at >= 0 {
		l.budget = document.Items[at].Budget
	}
	if err := l.shipSet(ctx); err != nil {
		return "", err
	}
	checks := []struct {
		name string
		run  func(context.Context) error
	}{
		{"format", l.format},
		{"fix", l.fix},
		{"vet", l.vet},
		{"rules", l.rules},
		{"test", l.test},
		{"lanes", l.lanes},
		{"verify", func(ctx context.Context) error { return l.verify(ctx, verify) }},
		{"scorecard", l.scorecard},
	}
	// A change to the plan alone is data the load above has validated;
	// nothing compiles it, so only the row's verify can speak to it, and
	// the tests that read the plan run with the next code landing.
	if !slices.ContainsFunc(l.paths, func(changed string) bool { return changed != plan.Path }) {
		checks = slices.DeleteFunc(checks, func(check struct {
			name string
			run  func(context.Context) error
		}) bool {
			return check.name != "verify"
		})
	}
	for _, check := range checks {
		started := time.Now()
		if err := check.run(ctx); err != nil {
			return "", fmt.Errorf("land: %s: %w", check.name, err)
		}
		wall := time.Since(started).Round(time.Millisecond)
		entry := check.name + " " + wall.String()
		if detail := l.details[check.name]; detail != "" {
			entry += " (" + detail + ")"
		}
		l.measured = append(l.measured, entry)
		fmt.Fprintf(options.Log, "land: %s ok %s\n", check.name, wall)
	}
	return l.commit(ctx)
}

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

func rowVerify(document plan.Plan, item, step string) string {
	for _, candidate := range document.Items {
		for _, s := range candidate.Steps {
			if candidate.ID == item && s.ID == step {
				return s.Verify
			}
		}
	}
	return ""
}

// shipSet is the dirty tree. An untracked top-level entry HEAD does not hold
// is refused: a new root belongs in a change of its own.
func (l *landing) shipSet(ctx context.Context) error {
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

// buildTags are the tags vet and fix read with, so every build-tagged test
// file is checked.
const buildTags = "modeltest,integration"

// fix applies go fix's modernizers to the changed packages, as format
// applies gofmt: a legacy form go fix can rewrite is rewritten, not refused.
// A directory whose files this platform's build excludes, or that does not
// compile, has no package to fix; vet reports the latter. Both rewrite
// whole packages, so a file they rewrote that the change left untouched
// joins the ship set here rather than reading as a tree that moved.
func (l *landing) fix(ctx context.Context) error {
	directories := l.goPackages()
	if len(directories) != 0 {
		listed, err := l.run(ctx, nil, "go", append([]string{"list", "-e", "-tags", buildTags, "-f", "{{if not .Error}}{{.ImportPath}}{{end}}"}, directories...)...)
		if err != nil {
			return err
		}
		if packages := strings.Fields(listed); len(packages) != 0 {
			if _, err := l.run(ctx, nil, "go", append([]string{"fix", "-tags", buildTags}, packages...)...); err != nil {
				return err
			}
		}
	}
	return l.shipSet(ctx)
}

func (l *landing) vet(ctx context.Context) error {
	_, err := l.run(ctx, nil, "go", "vet", "-tags", buildTags, "./...")
	return err
}

// test runs every package; go test's cache replays a package whose inputs
// did not change. No timeout is imposed: a test that must bound its own
// wait does so.
func (l *landing) test(ctx context.Context) error {
	output, err := l.run(ctx, nil, "go", "test", "-json", "-timeout=0", "./...")
	if err != nil {
		if summary := testevidence.FailureSummary(output); summary != "" {
			return fmt.Errorf("failed tests/packages: %s", summary)
		}
		return err
	}
	l.details["test"] = ranPackages(output)
	return nil
}

// reportedPackages is how many of the packages go test ran the measurement
// names: the slowest few are where a landing's test time went.
const reportedPackages = 5

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
		case parsed.Action == "output" && strings.Contains(parsed.Output, "\t(cached)"):
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

func (l *landing) verify(ctx context.Context, command string) error {
	if strings.TrimSpace(command) == "" {
		return nil
	}
	_, err := planverify.Execute(ctx, l.Repo, command, nil)
	return err
}

// commit writes the advanced plan and commits the tree the checks passed.
// HEAD and the dirty tree must be as the landing found them.
func (l *landing) commit(ctx context.Context) (string, error) {
	head, err := gitauthority.Query(ctx, l.Repo, "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(string(head)) != l.head {
		return "", errors.New("land: HEAD moved during the landing")
	}
	now, err := dirtyTree(ctx, l.Repo)
	if err != nil {
		return "", err
	}
	if !sameTree(now, l.entries) {
		return "", errors.New("land: the tree changed during the landing; land again once it is settled")
	}
	document, err := plan.Load(filepath.Join(l.Repo, plan.Path))
	if err != nil {
		return "", err
	}
	advanced, err := plan.Advance(document, l.item, l.step)
	if err != nil {
		return "", err
	}
	if err := plan.Save(filepath.Join(l.Repo, plan.Path), advanced); err != nil {
		return "", err
	}
	file, err := os.CreateTemp("", "overgo-landing-*.txt")
	if err != nil {
		return "", err
	}
	defer os.Remove(file.Name())
	if _, err := file.WriteString(l.message()); err != nil {
		file.Close()
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	if _, err := l.git(ctx, "add", "-A"); err != nil {
		return "", err
	}
	if _, err := l.git(ctx, "commit", "-q", "-F", file.Name()); err != nil {
		return "", err
	}
	landed, err := gitauthority.Query(ctx, l.Repo, "rev-parse", "HEAD")
	return strings.TrimSpace(string(landed)), err
}

// message is the authored message, what the gate measured, and the trailers:
// the scorecard slices, then the landed row.
func (l *landing) message() string {
	trailers := append(slices.Clone(l.trailers), plan.LandedRowTrailer+": "+l.Ref)
	return strings.TrimSpace(string(l.Message)) + "\n\nMeasured by the gate: " + strings.Join(l.measured, ", ") +
		"\n\n" + strings.Join(trailers, "\n") + "\n"
}

// sameTree compares two dirty-tree readings by path and status; the plan,
// which the commit itself rewrites, and a formatted file keep their paths.
func sameTree(now, then []repoanalysis.DirtyPath) bool {
	return slices.EqualFunc(now, then, func(a, b repoanalysis.DirtyPath) bool {
		return a.Path == b.Path && a.OriginalPath == b.OriginalPath
	})
}

func (l *landing) git(ctx context.Context, arguments ...string) (string, error) {
	return l.run(ctx, gitauthority.RepositoryEnvironment(), "git", gitauthority.WriterArguments(arguments...)...)
}

// run executes one command at the repository root and returns its standard
// output; a failure carries the tail of both streams.
func (l *landing) run(ctx context.Context, environment []string, name string, arguments ...string) (string, error) {
	var output, combined bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: name, Args: arguments, Dir: l.Repo, Env: environment,
		Stdout: io.MultiWriter(&output, &combined), Stderr: &combined,
	})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("%s %s: exit %d: %s", name, strings.Join(arguments, " "), receipt.ExitCode, tail(combined.String()))
	}
	return output.String(), err
}
