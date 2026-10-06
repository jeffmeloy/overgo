package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/scorecard"
)

// improvement closes the loop each time the plan empties: the scorecard run
// measures, each cell is compared with its newest earlier complete
// measurement, and the gaps return as rows of the plan at path
// plan. root is the checkout whose HEAD is measured and whose plan mutation
// lock is taken; records is the directory the scorecards are published in.
type improvement struct {
	root, plan, records string
	// measure runs the stale validations the scorecard compares and publishes
	// it; fill runs the stale ones it does not compare. Each runs
	// in the cycle checkout it is given.
	measure func(checkout string) error
	fill    func(checkout string) error
	// limit is how many landings that moved what a cycle measures start a cycle between
	// steps; zero starts none, so a cycle starts only when the plan empties and something moved.
	limit int
	// surface reads the packages a cycle's commands depend on; nil reads them with go list.
	surface func(ctx context.Context, root string) (map[string]bool, error)
	// watch says the loop gates its cycles on what moved: an empty plan with nothing moved since
	// the last cycle that completed starts none. The loop's world sets it; a bare improvement
	// runs a cycle whenever the plan empties, as before.
	watch bool
	// attempted is the commit a cycle the trigger started was begun at, so one that did not
	// publish is not started again for the same commit.
	attempted string
}

// movedSince is the commit of the newest scorecard published and how many commits since it
// moved what a cycle measures.
func (w *improvement) movedSince() (last string, moved int, err error) {
	ctx := context.Background()
	read := surfaceClosure
	if w.surface != nil {
		read = w.surface
	}
	closure, err := read(ctx, w.root)
	if err != nil {
		return "", 0, err
	}
	return landingsThatMovedTheSurface(ctx, w.root, w.records, closure)
}

// CycleDue says whether landings that moved what a cycle measures have reached the limit since
// the last cycle that completed, of any outcome, and no cycle was begun for this commit already.
func (w *improvement) CycleDue() (bool, string, error) {
	if w.limit <= 0 {
		return false, "no landing limit is set, so a cycle starts only when the plan empties", nil
	}
	head, err := gitauthority.Query(context.Background(), w.root, "rev-parse", "HEAD")
	if err != nil {
		return false, "", err
	}
	if commit := strings.TrimSpace(string(head)); commit == w.attempted {
		return false, "a cycle was begun at this commit and published nothing", nil
	}
	last, moved, err := w.movedSince()
	if err != nil {
		return false, "", err
	}
	decision := decideCycle(last, moved, w.limit)
	if decision.Due {
		w.attempted = strings.TrimSpace(string(head))
	}
	return decision.Due, decision.Reason, nil
}

// CycleWorthRunning says whether a cycle an empty plan would start has anything to measure: it has
// when no cycle has completed or something it measures has moved since the last that did.
func (w *improvement) CycleWorthRunning() (bool, string, error) {
	if !w.watch {
		return true, "the cycle is not gated on what moved", nil
	}
	last, moved, err := w.movedSince()
	if err != nil {
		return false, "", err
	}
	if last == "" || moved > 0 {
		return true, "what a cycle measures moved since the last cycle that completed", nil
	}
	return false, decideCycle(last, moved, w.limit).Reason, nil
}

// improvementFor is the loop's improvement world over a store: this
// checkout's plan, the store's published scorecards, the scorecard run and
// the background validations.
func improvementFor(store string) improvement {
	return improvement{
		watch: true, root: loopWorktree, plan: plan.Path, records: scorecard.Directory(store),
		measure: func(checkout string) error { return measureScorecard(checkout, store) },
		fill: func(checkout string) error {
			return validateIn(checkout, store, []string{cycleCreditEnvironment(filepath.Dir(checkout))}, "-run", "-background")
		},
	}
}

// Measure runs the scorecard at root's HEAD in the cycle checkout and
// returns the record published at that commit and the reference its cells are
// compared with (scorecard.Reference). A
// cycle the run before this one left unfinished continues where it stopped
// (see cycleState), and the record it published is the one returned. Rows land
// in root while it runs: the claims it records bind the commit the cycle
// checkout holds. A failed validation returns its error beside the record;
// a run that published nothing at the commit returns none.
func (w improvement) Measure() (*scorecard.Record, *scorecard.Record, error) {
	head, err := gitauthority.Query(context.Background(), w.root, "rev-parse", "HEAD")
	if err != nil {
		return nil, nil, err
	}
	commit := strings.TrimSpace(string(head))
	measured := w.inCycle(commit, w.measure)
	records, err := scorecard.Records(w.records)
	if err != nil || len(records) == 0 || records[len(records)-1].Commit != commit {
		return nil, nil, errors.Join(measured, err)
	}
	return &records[len(records)-1], scorecard.Reference(records), measured
}

// gitWrite runs one git command that writes the checkout at directory.
func gitWrite(directory string, arguments ...string) error {
	var output bytes.Buffer
	if err := runToolEnv(gitauthority.RepositoryEnvironment(), nil, &output, &output, append([]string{"git", "-C", directory}, gitauthority.WriterArguments(arguments...)...)...); err != nil {
		return fmt.Errorf("git %s: %w: %s", arguments[0], err, strings.TrimSpace(output.String()))
	}
	return nil
}

// Propose appends a row for each gap the plan does not hold, owned by the
// plan's lane and verified by the gap closing on the scorecard the store
// holds, and reports how many it added. A row that exists is never rewritten,
// so a decline's row keeps the value it was first to be restored to.
func (w improvement) Propose(published scorecard.Record, gaps []scorecard.Gap) (added int, err error) {
	lock, err := authoritylock.AcquireContext(context.Background(), w.root)
	if err != nil {
		return 0, err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	document, err := plan.Load(w.plan)
	if err != nil {
		return 0, err
	}
	for _, gap := range gaps {
		if slices.ContainsFunc(document.Items, func(item plan.Item) bool { return item.ID == gap.Key }) {
			continue
		}
		verify := "go run ./cmd/validate -scorecard -gap " + gap.Key
		if gap.Restore {
			verify += " -target " + strconv.FormatFloat(gap.Target, 'g', -1, targetBits)
		}
		document.Items = append(document.Items, plan.Item{ID: gap.Key, Title: gap.Title, Owner: document.Lane, Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: proposalStep, Title: gap.Title, Status: plan.StatusOpen, Verify: verify,
			Rationale: gap.Detail + "\nServes: measure and find the gap; the loop proposed this row from scorecard " + published.Name +
				". Evidence: the verifier reads the scorecard the store holds, compared with the measurements before it.",
		}}})
		added++
	}
	if added == 0 {
		return 0, nil
	}
	return added, plan.Save(w.plan, document)
}

// targetBits is the precision a decline's restoration target is written in, the float64 it is read as.
const targetBits = 64

// proposalStep is the one step a proposed row has.
const proposalStep = "do"

// Fill runs the background validations, the suites the baseline scorecard
// does not compare, after the cycle's decision: their evidence flows
// without delaying or withholding a promotion, and a failure is reported
// here and changes nothing the cycle decided. They run in the cycle
// checkout at the measured commit, as the scorecard did, so the claims they
// record bind that commit and rows land in the main checkout while they
// run.
func (w improvement) Fill(commit string) {
	// The cycle has reached its decision: the next one measures afresh.
	defer os.Remove(filepath.Join(w.root, processlock.StateDirectory, cycleStateFile))
	defer os.Remove(filepath.Join(w.root, processlock.StateDirectory, cycleCreditFile))
	err := w.inCycle(commit, func(checkout string) error {
		fmt.Printf("loop: filling the background validations at %.12s in %s\n", commit, checkout)
		return w.fill(checkout)
	})
	if err != nil {
		fmt.Printf("loop: the background validations failed: %v\n", err)
	}
}

// cycleCheckout is the detached worktree the cycle measures and fills in,
// kept in the process state of the checkout the loop runs in, with its lock
// beside it. It is the same directory every cycle, so go reuses the builds
// and test results it keys by package directory. The state directory
// ignores everything in it and go's ./... skips a directory whose name
// starts with a dot, so neither a landing nor a walk of cmd and internal
// ever reaches it.
const (
	cycleCheckout = "cycle"
	cycleLock     = cycleCheckout + ".lock"
	// cycleLockMode keeps the lock private to the checkout's owner.
	cycleLockMode = 0o600
)

// inCycle moves the cycle checkout to commit and runs run in it, holding
// the checkout's lock throughout: another cycle refuses rather than move
// the checkout under a run.
func (w improvement) inCycle(commit string, run func(checkout string) error) (err error) {
	if err := processlock.EnsureStateDirectory(w.root); err != nil {
		return err
	}
	state := filepath.Join(w.root, processlock.StateDirectory)
	lock, err := processlock.Acquire(filepath.Join(state, cycleLock), cycleLockMode)
	if err != nil {
		return fmt.Errorf("another cycle holds the cycle checkout: %w", err)
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	checkout, err := filepath.Abs(filepath.Join(state, cycleCheckout))
	if err != nil {
		return err
	}
	if err := w.moveCycle(checkout, commit); err != nil {
		return err
	}
	return run(checkout)
}

// moveCycle checks the cycle checkout out at commit, adding the worktree on
// first use. A checkout holding changes refuses, as does one holding an
// ignored file that commit tracks: a run left them, and nothing here
// discards them. Nothing removes the checkout either, so what a run leaves
// among its ignored files stays, as it does in the main checkout.
func (w improvement) moveCycle(checkout, commit string) error {
	if _, err := os.Lstat(filepath.Join(checkout, ".git")); errors.Is(err, fs.ErrNotExist) {
		return gitWrite(w.root, "worktree", "add", "--quiet", "--detach", checkout, commit)
	}
	status, err := gitauthority.Query(context.Background(), checkout, "status", "--porcelain")
	if err != nil {
		return err
	}
	if len(bytes.TrimSpace(status)) > 0 {
		return fmt.Errorf("the cycle checkout %s holds changes a run left; keep or remove them by hand:\n%s", checkout, bytes.TrimRight(status, "\n"))
	}
	return gitWrite(checkout, "checkout", "--quiet", "--detach", "--no-overwrite-ignore", commit)
}

// cycleState is how far the cycle for one key got, kept beside the cycle
// checkout: a run that ends before it finishes, the time limit of whoever
// started it among the causes, is continued by running the cycle again, which
// starts at the first phase not passed instead of acquiring every cell again.
// The key names what the credit is for: the commit, the store the cells and
// the scorecard go to, and the obligations the dry-run plan names, so a
// different store or an obligation that changed since (a model activated, a
// recipe replaced) starts over. Only a phase that passed is recorded, so a
// failure runs again; the lanes are not recorded, since go test's cache
// replays the packages whose inputs held and runs the rest.
type cycleState struct {
	Key       string `json:"key"`
	Cells     bool   `json:"cells"`
	Published bool   `json:"published"`
}

const (
	cycleStateFile = "cycle_state.json"
	cycleStateMode = 0o600
	// cycleCreditFile keeps the cells the cycle's runs acquired, beside the state, which
	// validate -run reads and writes through clioptions.CycleCreditEnvironment so a run
	// cut short continues from the cell it stopped at.
	cycleCreditFile = "cycle_credit.json"
)

// cycleCreditEnvironment is the setting that has a validate run of this cycle keep and read
// the cells it acquired in the state directory beside the cycle's checkout.
func cycleCreditEnvironment(stateDirectory string) string {
	return clioptions.CycleCreditEnvironment + "=" + filepath.Join(stateDirectory, cycleCreditFile)
}

// readCycleState is the state kept for key: none, for another key or when
// nothing is kept.
func readCycleState(path, key string) cycleState {
	var state cycleState
	if err := jsonfile.DecodeStrict(path, &state); err != nil || state.Key != key {
		return cycleState{Key: key}
	}
	return state
}

// obligationsDigest names what the dry-run plan asks of a cycle: the model,
// task, validation, protocol, recipe and location of every cell it runs or
// leaves to the background, the root its inventory was registered from and the
// committed data its training and projection runs read (grounding, objective,
// conditioning, projector), and why it is scheduled so, without what a run
// itself changes (the cost it measured, the record it replays).
func obligationsDigest(plan []byte) (string, error) {
	var decoded struct {
		Run        []map[string]any `json:"run"`
		Background []map[string]any `json:"background"`
	}
	if err := json.Unmarshal(plan, &decoded); err != nil {
		return "", err
	}
	var lines []string
	for bucket, cells := range map[string][]map[string]any{"run": decoded.Run, "background": decoded.Background} {
		for _, cell := range cells {
			parts := []string{bucket}
			for _, field := range []string{"model", "kind", "validation", "protocol", "recipe", "location", "directory", "grounding", "objective", "conditioning", "projector", "promotion"} {
				parts = append(parts, fmt.Sprint(cell[field]))
			}
			lines = append(lines, strings.Join(parts, "\x00"))
		}
	}
	slices.Sort(lines)
	sum := sha256.Sum256([]byte(strings.Join(lines, "\n")))
	return hex.EncodeToString(sum[:]), nil
}

// measureScorecard measures again, from checkout, the validations promotion
// waits on, then publishes the scorecard of store with what moved since the
// previous one, then runs the cycle lanes, which go test's cache replays for
// the packages no change reached. The scorecard publishes before the lanes, so
// the cells' numbers exist when a long lane run is cut short. A failed
// validation does not withhold the others' numbers: the scorecard publishes
// regardless and the failure is returned beside it with the final error the
// command wrote to its error file, so the withheld promotion names each cell
// or lane validate -run acquired nothing for, and why.
func measureScorecard(checkout, store string) error {
	head, err := gitauthority.Query(context.Background(), checkout, "rev-parse", "HEAD")
	if err != nil {
		return err
	}
	absolute, err := filepath.Abs(store)
	if err != nil {
		return err
	}
	var plan bytes.Buffer
	if err := validateTo(&plan, checkout, store, nil); err != nil {
		return fmt.Errorf("loop: the dry-run plan the cycle's phases are keyed by: %w", err)
	}
	obligations, err := obligationsDigest(plan.Bytes())
	if err != nil {
		return err
	}
	key := strings.Join([]string{strings.TrimSpace(string(head)), absolute, obligations}, "|")
	return measurePhases(filepath.Join(filepath.Dir(checkout), cycleStateFile), key, os.Stdout, func(environment []string, arguments ...string) error {
		return validateIn(checkout, store, environment, arguments...)
	})
}

// measurePhases runs the cycle's phases for key, the cells, the publication
// and the lanes, in that order, skipping the cells and the publication the
// state at statePath records as passed for that key. The cells run again
// whenever they did not pass, and the scorecard publishes again after them,
// since it then holds fresher numbers; the lanes always run. The report says
// what a resumed cycle credited and, for each phase it ran, how long it took
// and whether it passed.
func measurePhases(statePath, key string, report io.Writer, validate func(environment []string, arguments ...string) error) error {
	errorFile := filepath.Join(os.TempDir(), fmt.Sprintf("overgo-loop-measure-%d.error", os.Getpid()))
	defer os.Remove(errorFile)
	state := readCycleState(statePath, key)
	var failed []error
	phase := func(name string, arguments ...string) bool {
		_ = os.Remove(errorFile)
		started := time.Now()
		err := validate([]string{clioptions.ErrorFileEnvironment + "=" + errorFile, cycleCreditEnvironment(filepath.Dir(statePath))}, arguments...)
		verdict := "passed"
		if err != nil {
			verdict = "did not pass"
		}
		fmt.Fprintf(report, "loop: cycle phase %s %s in %s\n", name, verdict, time.Since(started).Round(time.Millisecond))
		if err != nil {
			final, _ := os.ReadFile(errorFile)
			failed = append(failed, fmt.Errorf("loop: the plan is empty, but validate %s failed: %w: %s", strings.Join(arguments, " "), err, bytes.TrimSpace(final)))
		}
		return err == nil
	}
	remember := func() {
		if err := jsonfile.Write(statePath, state, cycleStateMode); err != nil {
			failed = append(failed, err)
		}
	}
	remeasured := !state.Cells
	if remeasured {
		state.Cells = phase("cells", "-run", "-cells")
		remember()
	} else {
		fmt.Fprintln(report, "loop: cycle phase cells credited from the interrupted run of this key: not run again")
	}
	if remeasured || !state.Published {
		state.Published = phase("publication", "-scorecard", "-publish")
		remember()
	} else {
		fmt.Fprintln(report, "loop: cycle phase publication credited from the interrupted run of this key: not run again")
	}
	phase("lanes", "-run", "-lanes")
	return errors.Join(failed...)
}

// validateIn runs cmd/validate with arguments from checkout into store,
// with environment added. The checkout holds none of the main checkout's
// ignored files, so the data roots the main checkout resolves are bound for
// it explicitly.
func validateIn(checkout, store string, environment []string, arguments ...string) error {
	return validateTo(os.Stdout, checkout, store, environment, arguments...)
}

// validateTo is validateIn with the command's output sent to output.
func validateTo(output io.Writer, checkout, store string, environment []string, arguments ...string) error {
	store, err := filepath.Abs(store)
	if err != nil {
		return err
	}
	_, bound, err := dataroot.ResolveEnvironment(loopWorktree)
	if err != nil {
		return err
	}
	command := append([]string{"go", "-C", checkout, "run", "./cmd/validate"}, arguments...)
	return runToolEnv(slices.Concat(loopEnvironment(), bound, environment), nil, output, os.Stderr, append(command, "-repo", store)...)
}
