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
// The landing command's checks open by reading free space, and refuse to
// start below a floor (space).
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
	"overgo/internal/quietmachine"
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
	// Free reads the bytes free on the volume that holds a directory. With it
	// the checks open with the stage space, which refuses a run below the
	// floor. The landing command gives VolumeFree; a fixture gives none, so a
	// test of the gate is held to no floor of the machine it runs on.
	Free func(directory string) (uint64, error)
	// Dispatch holds the landing to what dispatch would give the role it names, under the plan the change carries: a
	// recorded stop, a step that waits on a prerequisite or a decision, another lane's row, and, for a worker, any step
	// but the dispatched one are refused (plan.AdmitLanding). The landing command sets it from its environment; a
	// fixture that tests something else leaves it off.
	Dispatch *plan.DispatchRequest
}

type landing struct {
	Options
	// head is the commit the landing found and publishes onto; merging are
	// the heads a merge in progress joins to it.
	head    string
	merging []string
	// scratch holds the private index the candidate is written through.
	scratch string
	entries []repoanalysis.DirtyPath
	paths   []string
	// written are the paths the change itself wrote, before the format and fix stages rewrote any (undo).
	written []string
	// scope is what the commit may hold (commitScope), nil for the whole tree; scoped says it was read, left the changed
	// paths it leaves out, and leftNamed that they were named (nameLeft).
	scope             []string
	left              []string
	scoped, leftNamed bool
	measured          []string
	// details say what a stage measured beyond its time: the packages the
	// test stage ran rather than replayed, what the scorecard slice found.
	details map[string]string
	// trailers are the scorecard slices' commit trailers, one per slice
	// that measured or failed.
	trailers []string
	// added are the checks of the steps this change itself adds.
	added []string
	// changedTerms are the trailers that state each row's check and count this change alters.
	changedTerms []string
	// trunk is the directory of the repository's main worktree and branch the
	// branch it has checked out; onto is that branch when this checkout is
	// not the one that has it, and the commit publishes onto it.
	trunk, branch, onto string
	// moved says what the trunk's plan changed of the rows' terms between
	// the commit the change was written on and the tip it was brought up to.
	moved string
	// quiet releases the landing's side of the quiet machine named quietMachine, nil while it holds none
	// (inLaneTurn).
	quiet        func() error
	quietMachine string
	// candidate is the tree the checks validate and the commit publishes.
	candidate string
	// turn gives back the repository's commit turn while the run holds it (holdTurn); letGo gives it back for a wait
	// on the device, and turnLet says the lanes must take it again before anything is published (takeTurnBack).
	turn    func() error
	turnLet bool
	// tree is the worktree of its own the candidate is checked out in, where the generators derive from it and every
	// stage after format and fix reads it (checkOut); treeHolds is the candidate it holds, and treeHeld gives the
	// worktree back for the next landing.
	tree      string
	treeHolds string
	treeHeld  func() error
	// planned is the plan the candidate holds and advanced is it with the
	// landed steps removed. terms is the plan the change is judged by,
	// counts the candidate's rows' counts, read before the landed steps leave
	// it, and verifiers the landed steps' checks as it states them.
	planned   []byte
	advanced  plan.Plan
	terms     plan.Plan
	counts    map[string]plan.Budget
	verifiers []string
	// spent is where the run's time went, for the checkout's record.
	spent spent
	// composed is the message the landing wrote for the change (compose), before what the gate measured; undone is
	// what the change's checks did with its production Go restored (undo).
	composed string
	undone   []string
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
	if err := creditsNoCoAuthor(options.Message); err != nil {
		return "", err
	}
	scratch, err := os.MkdirTemp("", "overgo-landing-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(scratch)
	l := &landing{Options: options, scratch: scratch, details: map[string]string{}}
	// The worktree the stages read the candidate in is kept for the next landing, and given back.
	defer func() {
		if l.treeHeld != nil {
			if released := l.treeHeld(); released != nil {
				fmt.Fprintf(options.Log, "land: the worktree %s the stages read is still held: %v\n", l.tree, released)
			}
		}
	}()
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
		if l.turn, err = l.holdTurn(ctx); err != nil {
			return "", err
		}
		defer func() {
			if l.turn != nil {
				_ = l.turn()
			}
		}()
		// Under the turn the trunk stands still: a worktree that is behind it is brought up to it.
		if err := l.integrate(ctx); err != nil {
			return "", err
		}
	}
	if options.Quiet {
		release, err := l.holdQuiet(ctx, quietmachine.Name)
		if err != nil {
			return "", err
		}
		l.quiet, l.quietMachine = release, quietmachine.Name
		defer func() {
			if l.quiet != nil {
				_ = l.quiet()
			}
		}()
	}
	if err := l.hooked(ctx); err != nil {
		return "", err
	}
	if err := l.parents(ctx); err != nil {
		return "", err
	}
	// refused adds, to a refusal, the paths the commit left out and the command that lists them (leftOut), and, to the
	// refusal of a change brought up to a trunk whose plan moved its rows' terms, what moved.
	refused := func(err error) error {
		if note := l.leftOut(); note != "" {
			err = fmt.Errorf("%w\n%s", err, note)
		}
		if l.moved == "" {
			return err
		}
		return fmt.Errorf("%w\nland: since this change was written the trunk's plan moved its terms: %s", err, l.moved)
	}
	if err := l.shipSet(ctx); err != nil {
		if landed := l.landedAs(ctx); landed != "" {
			fmt.Fprintf(options.Log, "land: %s had landed as %.12s, and nothing of it is left here to commit\n", options.Ref, landed)
			l.spent.Outcome = outcomeLanded
			return landed, nil
		}
		return "", refused(err)
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
	// Free space is read first whatever the run keeps: a preflight and a
	// change to the plan alone build too.
	if l.Free != nil {
		checks = slices.Insert(checks, 0, check{spaceClass, l.space, true})
	}
	for _, check := range checks {
		if readsCandidate(check.name) {
			if err := l.checkOut(ctx); err != nil {
				return "", fmt.Errorf("%scandidate worktree: %w", stagePrefix, err)
			}
		}
		started := time.Now()
		l.spent.Outcome = check.name
		err := check.run(ctx)
		wall := time.Since(started).Round(time.Millisecond)
		if l.spent.Stages[check.name] = wall; err != nil {
			return "", refused(fmt.Errorf("%s%s: %w", stagePrefix, check.name, err))
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

// creditsNoCoAuthor refuses the lead's words when a line of them credits a co-author: a landing's message carries none.
// The landing writes the rest of the message itself (compose).
func creditsNoCoAuthor(message []byte) error {
	for line := range strings.Lines(string(message)) {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(line)), "co-authored-by:") {
			return errors.New("land: the message credits a co-author; a landing's message carries none")
		}
	}
	return nil
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
	if !l.scoped {
		if l.scope, err = l.commitScope(ctx); err != nil {
			return err
		}
		l.scoped = true
	}
	if l.candidate, err = l.capture(ctx); err != nil {
		return err
	}
	changed, err := dirtyTree(ctx, l.Repo)
	if err != nil {
		return err
	}
	entries, left := inside(l.scope, changed)
	l.nameLeft(left)
	l.paths = nil
	for _, entry := range entries {
		l.paths = append(l.paths, entry.Path)
		if entry.OriginalPath != "" {
			l.paths = append(l.paths, entry.OriginalPath)
		}
	}
	// The first reading is what the change wrote, before the format and fix stages rewrite anything (undo).
	if l.written == nil {
		l.written = l.paths
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

// findingsRegister is the register a review session writes its findings into: a record no code reads, as the plan is
// one no code compiles, and one the guard keeps a review session from committing, so it waits for the lead's next
// landing.
const findingsRegister = plan.FindingsRegister

// planOnly reports a change to the plan alone, or to the plan and the findings register a review left uncommitted.
func (l *landing) planOnly() bool {
	return !slices.ContainsFunc(l.paths, func(changed string) bool { return changed != plan.Path && changed != findingsRegister })
}

// capture writes the working tree, as git add -A stages it, through the
// landing's private index and returns its tree; a scoped landing stages only
// the changed paths its scope holds, so whatever else the checkout holds stays
// as it stands and outside the commit. The index the checkout's own git
// commands use is neither read nor written.
func (l *landing) capture(ctx context.Context) (string, error) {
	arguments := []string{"add", "-A"}
	if l.scope != nil {
		changed, err := dirtyTree(ctx, l.Repo)
		if err != nil {
			return "", err
		}
		held, _ := inside(l.scope, changed)
		arguments = append(arguments, "--")
		for _, entry := range held {
			arguments = append(arguments, entry.Path)
			if entry.OriginalPath != "" {
				arguments = append(arguments, entry.OriginalPath)
			}
		}
	}
	if !slices.Equal(arguments, []string{"add", "-A", "--"}) {
		if _, err := l.staged(ctx, arguments...); err != nil {
			return "", err
		}
	}
	tree, err := l.staged(ctx, "write-tree")
	return strings.TrimSpace(tree), err
}

// row reads the plan the candidate holds, which the commit publishes with the
// landed steps removed, and the terms the change is judged by: each landed
// step's check and what its row may still add. The terms come from the plan
// of the commit the landing builds on, so a change that edits its own row's
// check is held to the one it started with, a count it raises counts only
// with a reason of its own (charge), and a step the change itself adds has its
// own, its check failing without the change (addedChecksFail). Two changes have no earlier terms and
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
	held, heldErr := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.head+":"+plan.Path)
	if len(l.merging) == 0 && !l.planOnly() {
		if heldErr != nil {
			return heldErr
		}
		if l.terms, err = plan.Parse(held); err != nil {
			return err
		}
	}
	// A plan the commit it builds on does not hold has no terms to change.
	var replaced []string
	var before plan.Plan
	if heldErr == nil {
		if before, err = plan.Parse(held); err != nil {
			return err
		}
		l.changedTerms = termsChanged(before, document)
		if err := l.landedAgain(ctx, before, document); err != nil {
			return err
		}
		// A merge's removed steps are the other line's landings, carried with their own trailers.
		if len(l.merging) == 0 {
			var removed []string
			removed, replaced = rowChanges(before, document)
			for _, row := range removed {
				retired := slices.ContainsFunc(document.Retired, func(record plan.Retirement) bool {
					return row == record.Row || strings.HasPrefix(row, record.Row+"/")
				})
				if !retired && !strings.Contains(string(l.Message), row) {
					return fmt.Errorf("land: the change removes %s from the plan without landing it: retire it with go run ./cmd/plan -retire -reason <why> %s", row, row)
				}
				l.changedTerms = append(l.changedTerms, retiredRowTrailer+": "+row)
			}
		}
	}
	named, err := rows(l.Ref)
	if err != nil {
		return err
	}
	if l.Dispatch != nil {
		refs := make([]plan.Ref, len(named))
		for index, row := range named {
			refs[index] = plan.Ref{Item: row.item, Step: row.step}
		}
		if err := plan.AdmitLanding(ctx, l.Repo, document, *l.Dispatch, refs); err != nil {
			return fmt.Errorf("land: %w", err)
		}
	}
	l.advanced, l.verifiers, l.added = document, nil, replaced
	l.counts = make(map[string]plan.Budget, len(document.Items))
	for _, item := range document.Items {
		l.counts[item.ID] = item.Budget
	}
	for _, row := range named {
		if step, judged := plan.ExactStep(l.terms, row.item, row.step); !judged {
			if err := l.judgeAdded(document, row); err != nil {
				return err
			}
		} else if !slices.Contains(l.verifiers, step.Verify) {
			l.verifiers = append(l.verifiers, step.Verify)
		}
		if l.advanced, err = plan.Advance(l.advanced, row.item, row.step); err != nil {
			return fmt.Errorf("land: %s cannot leave the plan: %w", row, err)
		}
	}
	// The message says the retirements, so the plan the commit publishes holds them no more.
	l.advanced.Retired, l.advanced.Answered, l.planned = nil, nil, data
	l.composed, err = l.compose(ctx, before, document, named)
	return err
}

// landedAgain refuses a change whose plan holds a step the trunk's plan does not, which a landing on the trunk
// already removed: its plan is from before that landing, as a landing killed between its ref's move and its plan's
// save leaves the checkout's, and publishing it would file the landed step again. A step retired is no landing, so a
// retired step filed again is held to nothing here.
func (l *landing) landedAgain(ctx context.Context, before, document plan.Plan) error {
	held := map[string]bool{}
	for _, item := range before.Items {
		for _, step := range item.Steps {
			held[item.ID+"/"+step.ID] = true
		}
	}
	for _, item := range document.Items {
		for _, step := range item.Steps {
			if held[item.ID+"/"+step.ID] {
				continue
			}
			// The search finds the trailer's text anywhere in a message; the step landed only where a parsed trailer
			// names it whole, so a step whose name merely contains this one's, or prose quoting it, is no landing.
			row := item.ID + "/" + step.ID
			found, err := gitauthority.Query(ctx, l.Repo, "log", "--fixed-strings", "--grep="+plan.LandedRowTrailer+": "+row,
				"--format=%H %(trailers:key="+plan.LandedRowTrailer+",valueonly,separator=%x1f)", l.builtOn())
			if err != nil {
				return err
			}
			for line := range strings.Lines(string(found)) {
				landed, values, _ := strings.Cut(strings.TrimSpace(line), " ")
				if !slices.Contains(strings.Split(values, trailerSeparator), row) {
					continue
				}
				return fmt.Errorf("land: the change's plan holds %s/%s, which landed as %.12s: the plan is from before that landing, as a landing killed before it saved its plan leaves it; take the trunk's plan (git checkout HEAD -- %s), make this change's own plan edits again, and file new work under a new step", item.ID, step.ID, landed, plan.Path)
			}
		}
	}
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
//   - a landed row whose count the change's own plan raises, with a reason of its own, is admitted by the raised
//     count, as a landing of the plan alone before it would admit it, with no gate run of its own
func (l *landing) charge(growth int) (left int) {
	named, _ := rows(l.Ref)
	for _, item := range l.terms.Items {
		if !slices.ContainsFunc(named, func(row ref) bool { return row.item == item.ID }) {
			continue
		}
		count := item.Budget.Nodes
		if raised := l.counts[item.ID]; raised.Nodes > count && raised.Reason != "" && raised.Reason != item.Budget.Reason {
			count = raised.Nodes
		}
		if at := slices.IndexFunc(l.advanced.Items, func(open plan.Item) bool { return open.ID == item.ID }); at >= 0 {
			l.advanced.Items[at].Budget.Nodes, growth = count-growth, 0
		}
		left += count
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
		listed, err := l.run(ctx, nil, "go", append([]string{"list", "-e", "-tags", checkedTags, "-f", "{{if not .Error}}{{.ImportPath}}{{end}}"}, directories...)...)
		if err != nil {
			return err
		}
		if packages := strings.Fields(listed); len(packages) != 0 {
			// go fix keeps what it fixed under the names of the files it was given, in an entry whose
			// key forgets the directory once a build is trimmed. Another checkout of the same bytes
			// would then have the first checkout's files written and its own left as they were. So
			// this one command is never trimmed, whatever the environment asks of the others.
			if _, err := l.run(ctx, nil, "go", append([]string{"fix", "-trimpath=false", "-tags", checkedTags}, packages...)...); err != nil {
				return err
			}
		}
		// A renamed test's citations follow it (followRenames), as go fix is applied.
		if err := l.followRenames(ctx); err != nil {
			return err
		}
		// What the source derives is regenerated from the candidate (derive), as go fix is applied, not refused for a
		// landing again after.
		if generators := l.generators(); len(generators) != 0 {
			if err := l.derive(ctx, generators); err != nil {
				return err
			}
		}
	}
	return l.shipSet(ctx)
}

// generator is a command that rewrites the files the source derives, which writes names.
type generator struct {
	arguments, environment, writes []string
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
		{"api-manifest", generator{arguments: []string{"run", "./cmd/api-manifest", "-update"}, writes: []string{"docs/api_manifest.json"}}},
		{"validate", generator{arguments: []string{"run", "./cmd/validate", "-claims", "-run"}, environment: append(os.Environ(), cudaTestEnvironment),
			writes: []string{"compatibility.json", "docs/COMPATIBILITY.md", "docs/TRAINING_COMPATIBILITY.md"}}},
	} {
		if _, err := os.Stat(filepath.Join(l.Repo, "cmd", candidate.command)); err == nil {
			held = append(held, candidate.generator)
		}
	}
	return held
}

func (l *landing) vet(ctx context.Context) error {
	_, err := l.runIn(ctx, l.stageRoot(), nil, "go", "vet", "-tags", checkedTags, "./...")
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
		_, err := os.Stat(filepath.Join(l.stageRoot(), filepath.FromSlash(pattern)))
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
	output, err := l.runIn(ctx, l.stageRoot(), nil, "go", append([]string{"test", "-json", "-timeout=0"}, packages...)...)
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
		if _, err := planverify.Execute(ctx, l.stageRoot(), verifier, nil); err != nil {
			return err
		}
	}
	if err := l.addedChecksFail(ctx); err != nil {
		return err
	}
	return l.undo(ctx)
}

// judgeAdded takes a step the change itself adds: nothing earlier states its terms, so its own check and count judge it.
func (l *landing) judgeAdded(document plan.Plan, row ref) error {
	for _, item := range document.Items {
		for _, step := range item.Steps {
			if item.ID != row.item || step.ID != row.step {
				continue
			}
			l.added = append(l.added, step.Verify)
			if !slices.Contains(l.verifiers, step.Verify) {
				l.verifiers = append(l.verifiers, step.Verify)
			}
			if !slices.ContainsFunc(l.terms.Items, func(held plan.Item) bool { return held.ID == item.ID }) {
				l.terms.Items = append(slices.Clone(l.terms.Items), item)
			}
			return nil
		}
	}
	return fmt.Errorf("land: %s is not a step of the plan this change carries", row)
}

// addedChecksFail runs an added step's check on the commit the change builds on, where it must fail: that keeps the proof a
// check filed before its work gives, that the check speaks of the change.
func (l *landing) addedChecksFail(ctx context.Context) error {
	if len(l.added) == 0 {
		return nil
	}
	tree, release, err := l.treeAt(ctx, "base")
	if err != nil {
		return err
	}
	defer release()
	for _, check := range l.added {
		if _, err := planverify.Execute(ctx, tree, check, nil); err == nil {
			return fmt.Errorf("land: the check %q passes on %.12s without this change, so it proves nothing of it", check, l.head)
		}
	}
	return nil
}

// regularFile is the Git mode the advanced plan is staged with.
const regularFile = "100644"

// settled holds the commit to the candidate the checks passed, which the stages read in a worktree of its own
// (checkOut): the commit publishes it whatever the checkout holds by then, which is left as edited.
func (l *landing) settled(ctx context.Context) error {
	_, err := l.staged(ctx, "read-tree", l.candidate)
	return err
}

// commit publishes the candidate the checks passed, its plan advanced past
// the row, onto the HEAD the landing found. What the checkout holds by then
// stays out of it (settled), and HEAD moves by
// one compare-and-set, so a refusal writes nothing: the plan keeps its row
// and any edit made meanwhile. Only a landed commit advances the working
// plan, and only while that plan is still the candidate's; a plan edited
// during the landing is left as edited and the landing says so.
func (l *landing) commit(ctx context.Context) (landed string, err error) {
	lock, err := authoritylock.AcquireContext(ctx, l.Repo)
	if err != nil {
		return "", err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	if err := l.settled(ctx); err != nil {
		return "", err
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
	// first, and only if its checkout can follow with what the lead has
	// uncommitted there kept (trunkFollows); the checkout follows next, and
	// the worktree's own HEAD moves last, so a commit stopped anywhere after
	// the trunk moved is found on it when begun again.
	var followed error
	if l.onto != "" {
		follow, err := l.trunkFollows(ctx, landed)
		if err != nil {
			return "", err
		}
		if _, err := l.moveRef(ctx, "-m", "land "+l.Ref, l.onto, landed, l.head); err != nil {
			return "", errors.Join(fmt.Errorf("land: publish %s onto the trunk %s at %.12s: %w", l.Ref, l.onto, l.head, err), follow(false))
		}
		if followed = follow(true); followed != nil {
			followed = fmt.Errorf("land: %s landed as %.12s, but the trunk's checkout %s did not follow it and holds the files of %.12s: %w", l.Ref, landed, l.trunk, l.head, followed)
		}
	}
	if _, err := l.moveRef(ctx, "-m", "land "+l.Ref, "HEAD", landed, l.head); err != nil {
		return "", errors.Join(fmt.Errorf("land: publish %s onto %.12s: %w", l.Ref, l.head, err), followed)
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

// message is the composed message, what the undo run found, what the gate measured, and the trailers:
// the scorecard slices, then the landed row.
func (l *landing) message() string {
	trailers := slices.Concat(l.trailers, l.changedTerms)
	named, _ := rows(l.Ref)
	for _, row := range named {
		trailers = append(trailers, plan.LandedRowTrailer+": "+row.String())
	}
	return l.composed + l.section(undoSection, l.undone) + "\n\nMeasured by the gate: " + strings.Join(l.measured, ", ") +
		"\n\n" + strings.Join(trailers, "\n") + "\n"
}

func (l *landing) git(ctx context.Context, arguments ...string) (string, error) {
	return l.run(ctx, gitauthority.RepositoryEnvironment(), "git", gitauthority.WriterArguments(arguments...)...)
}

// moveRef moves a ref as a landing, which the trunk's hook admits.
func (l *landing) moveRef(ctx context.Context, arguments ...string) (string, error) {
	environment := append(gitauthority.RepositoryEnvironment(), trunkMoveEnvironment+"=1")
	return l.run(ctx, environment, "git", gitauthority.WriterArguments(append([]string{"update-ref"}, arguments...)...)...)
}

// hooked refuses a repository that carries the trunk's hook with the hook off: nothing in a checkout sets core.hooksPath, so
// a clone runs without it until a landing says so. The setting is read as git reads it, a relative path from the
// repository's root, and holds the hook exactly when it names the hook's folder, however it is written.
func (l *landing) hooked(ctx context.Context) error {
	folder := filepath.Join(l.Repo, filepath.FromSlash(trunkHookDirectory))
	if _, err := os.Stat(filepath.Join(folder, trunkHook)); err != nil {
		return nil
	}
	set, _ := gitauthority.Query(ctx, l.Repo, "config", "core.hooksPath")
	named := strings.TrimSpace(string(set))
	if !filepath.IsAbs(named) {
		named = filepath.Join(l.Repo, named)
	}
	hooks, namedErr := os.Stat(named)
	want, wantErr := os.Stat(folder)
	if namedErr != nil || wantErr != nil || !os.SameFile(hooks, want) {
		return fmt.Errorf("land: the trunk's hook is off, so a raw commit could move it: run git config core.hooksPath %s", trunkHookDirectory)
	}
	return nil
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
	return l.runIn(ctx, l.Repo, environment, name, arguments...)
}

// stageRoot is the directory the stages read the candidate in: the worktree it was checked out in once the checkout
// was let go (letGo), or the checkout.
func (l *landing) stageRoot() string { return cmp.Or(l.tree, l.Repo) }

// runIn is run in the directory named.
func (l *landing) runIn(ctx context.Context, directory string, environment []string, name string, arguments ...string) (string, error) {
	// Every command says whose it is, so a measurement it runs does not wait for this landing.
	if environment == nil {
		environment = os.Environ()
	}
	environment = append(slices.Clone(environment), quietmachine.HeldEnvironment+"=1")
	var output, diagnostics bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: name, Args: arguments, Dir: directory, Env: environment, Stdout: &output, Stderr: &diagnostics,
	})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("%s %s: exit %d: %s", name, strings.Join(arguments, " "), receipt.ExitCode, clioptions.Tail(output.String()+diagnostics.String(), clioptions.DiagnosticTailBytes))
	}
	return output.String(), err
}
