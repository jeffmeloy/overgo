package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/landing"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
	"overgo/internal/testskip"
)

// Landing a row is a fixed sequence around the one thing a worker authors:
// the small gate lands the row over the dirty tree, and the landing is
// believed only from git. The sequence is code so that no step can be
// skipped by memory, and every run ends in one typed outcome.

// Row sequence phases, in order.
const (
	phaseGate   = "gate"
	phaseLanded = "landed"
)

// Row sequence outcomes.
const (
	// outcomeRefused stops before or at the gate: nothing landed.
	outcomeRefused = "refused"
	// outcomeExternal stops at the gate for a cause outside the change:
	// nothing landed, and the refusal names the probe that re-checks it.
	outcomeExternal = "external-prerequisite"
	// outcomeReady landed, and git confirms it.
	outcomeReady = "ready"
)

// landOutcome is the typed result of one row sequence.
type landOutcome struct {
	Reference string `json:"reference"`
	Phase     string `json:"phase"`
	Outcome   string `json:"outcome"`
	Commit    string `json:"commit,omitzero"`
	Detail    string `json:"detail,omitzero"`
	// Refusal is a refusal's class as the checkout's record holds it.
	Refusal *landing.Refusal `json:"refusal,omitempty"`
	// Cleared says what a confirmed landing's clearing removed, or why it could not.
	Cleared string `json:"cleared,omitzero"`
	// Lanes says, of a confirmed landing that closed a row, where and when the cycle's lanes run at its commit
	// (lanesOnClose).
	Lanes string `json:"lanes,omitzero"`
}

// rowWorld is what the row sequence drives. A step that refuses returns its
// reason as text; an error is the step being unable to run at all. Cleared
// removes what nothing holds and says what it did: it has no failure to give.
type rowWorld interface {
	Gate(reference, messageFile string) (failure string, err error)
	Landed(reference string) (commit string, err error)
	Cleared(reference string) string
}

// landRow runs the sequence and stops at the first phase that does not pass.
// A landing git confirms clears, once; no other does, and the clearing changes
// nothing of the outcome but what it says of itself.
func landRow(world rowWorld, reference, messageFile string) (landOutcome, error) {
	outcome := landOutcome{Reference: reference, Outcome: outcomeRefused, Phase: phaseGate}
	refusal, err := world.Gate(reference, messageFile)
	if err != nil || refusal != "" {
		outcome.Detail = refusal
		return outcome, err
	}
	outcome.Phase = phaseLanded
	commit, err := world.Landed(reference)
	if err != nil {
		return outcome, err
	}
	outcome.Commit, outcome.Outcome, outcome.Cleared = commit, outcomeReady, world.Cleared(reference)
	return outcome, nil
}

// execRowWorld runs the sequence against the repository's real owners. Its
// subprocesses keep the caller's execution mode. The message is the gate's
// standard input when the worker gave it on the landing's.
type execRowWorld struct {
	message []byte
	// configured is what the loop's configuration sets, read once, by the landing's first tool (tool).
	configured map[string]string
	read       bool
	// reported is the commit the gate said the row landed as.
	reported string
}

// landedAs opens the commit in the gate's line that says a row landed.
const landedAs = " landed as "

// tool runs one repository tool, echoing its standard output -- a tool's
// verdict -- to echo and its standard error -- its progress -- to said as it
// comes, and returns a failed run's refusal: the final error the tool wrote to
// its error file, or, when it wrote none, as a build that never ran it or a
// tree its bound ended, how it ended and the tail of its standard error
// (endedWithout).
func (w *execRowWorld) tool(echo, said io.Writer, arguments ...string) (string, error) {
	errorFile := filepath.Join(os.TempDir(), fmt.Sprintf("overgo-loop-%d.error", os.Getpid()))
	_ = os.Remove(errorFile)
	// A failed run's file is read below and is then nobody's: it does not outlive the run.
	defer os.Remove(errorFile)
	progress := clioptions.NewTailWriter(clioptions.DiagnosticTailBytes)
	// The configuration is read once, at the first tool, the checker's build, before any check: one that cannot be read
	// refuses the landing there, where it was read as none at each tool and the landing went on without what it names
	// (SQA-380).
	if !w.read {
		configured, err := configuredEnvironment()
		if err != nil {
			return err.Error(), err
		}
		w.configured, w.read = configured, true
	}
	err := runToolEnv(append(landingEnvironment(os.Environ(), w.configured), clioptions.ErrorFileEnvironment+"="+errorFile), bytes.NewReader(w.message), echo, io.MultiWriter(progress, said), arguments...)
	if err == nil {
		return "", nil
	}
	final, _ := os.ReadFile(errorFile)
	if refusal := strings.TrimSpace(string(final)); refusal != "" {
		return refusal, err
	}
	return endedWithout(filepath.Base(arguments[0]), err, progress.Tail()), err
}

// endedWithout is the refusal of a tool that ended without writing one (verdict-steady): its first line, the one a
// landing's record keeps (landing.Classify), says how the run ended, the bound that stopped its tree or the status it
// ended with, and the last line the tool wrote, and its last output follows. The commit attempt for
// lanes-package-pass/do at 09:31 on 2026-11-19 ended in its test stage with no error file, and its record kept the
// first line of its output alone, a note on files left outside the commit, the run's error dropped.
func endedWithout(tool string, err error, tail string) string {
	tail = strings.TrimSpace(tail)
	last := tail[strings.LastIndexByte(tail, '\n')+1:]
	return fmt.Sprintf("%s ended without its refusal (%s); it last wrote: %s\n%s",
		tool, strings.ReplaceAll(err.Error(), "\n", "; "), cmp.Or(strings.TrimSpace(last), "nothing"), tail)
}

// checkerCheckout is the worktree under the repository's process state the checker is built in, beside the replay's.
const checkerCheckout = "checker"

// buildChecker builds command, the gate or the lanes runner (startLanesRunner), into output from the trunk's commit
// alone, in a worktree of its own moved there by git (moveTree), one build at a time, so a commit is checked by
// committed code: built in the checkout, the uncommitted files of the landing, the plan or process control changed how
// a commit was checked, and 3e7b49d0c was checked with uncommitted counts code (clean-checker/do). Building reads no
// data root, so none is copied in. A checker that cannot be set up refuses with what stopped it, as one that does not
// build refuses with the build's output: Gate took an empty refusal for a checker that ran, and its landing went on to
// confirm a commit that was never made (SQA-397).
func buildChecker(root, output, command string, build func(arguments ...string) (string, error)) (refusal string, err error) {
	defer func() {
		if err != nil && refusal == "" {
			refusal = err.Error()
		}
	}()
	head, err := gitauthority.Query(context.Background(), root, "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return "", err
	}
	state := filepath.Join(root, processlock.StateDirectory)
	lock, err := processlock.AcquireContext(context.Background(), filepath.Join(state, checkerCheckout+".lock"), clioptions.PrivateFileMode)
	if err != nil {
		return "", err
	}
	defer lock.Close()
	tree, err := filepath.Abs(filepath.Join(state, checkerCheckout))
	if err != nil {
		return "", err
	}
	commit := strings.TrimSpace(string(head))
	if err := moveTree(root, tree, commit); err != nil {
		return "", err
	}
	// The build is told the commit it was built from (builtCommit): go stamps a build from the git root it finds above
	// its module, and takes only a directory named .git for one, so in this worktree, whose .git is a file, it stamped
	// the main checkout's HEAD and uncommitted changes, and the runner said "with its tree's changes" of a clean build
	// (runner-from-commit).
	return build("go", "-C", tree, "build", "-ldflags=-X main.builtCommit="+commit, "-o", output, command)
}

// Gate lands the row through the small gate, built from the tree it lands
// into a directory of its own: a fixed path collided with the previous
// landing's gate while Windows still released its executable, and the build
// refused the half-released file.
func (w *execRowWorld) Gate(reference, messageFile string) (string, error) {
	directory, err := os.MkdirTemp("", "overgo-gate-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(directory)
	gate := filepath.Join(directory, "gate.exe")
	refusal, err := buildChecker(loopWorktree, gate, "./cmd/gate", func(arguments ...string) (string, error) {
		return w.tool(io.Discard, io.Discard, arguments...)
	})
	if err == nil {
		var said bytes.Buffer
		arguments := []string{gate, "-plan", reference}
		if messageFile != "" {
			arguments = append(arguments, "-message-file", messageFile)
		}
		// The gate's progress is said as it comes, a wait for the card's turn among it, whom it waits for and since
		// when: a landing's output had said nothing for the 6.8 hours its lanes waited (card-handover/lane-turn).
		refusal, _ = w.tool(io.MultiWriter(os.Stderr, &said), os.Stderr, arguments...)
		if _, commit, found := strings.Cut(said.String(), landedAs); found {
			w.reported, _, _ = strings.Cut(commit, "\n")
		}
	}
	return refusal, nil
}

// Landed confirms from git that the commit the gate reported, or HEAD when it
// reported none, is the landing of reference and that HEAD holds it. It is
// HEAD itself for a commit just made; for one the trunk already held, which a
// worktree brought up to the trunk finds, it is a commit HEAD descends from.
func (w *execRowWorld) Landed(reference string) (string, error) {
	named := cmp.Or(strings.TrimSpace(w.reported), "HEAD")
	message, err := gitauthority.Query(context.Background(), loopWorktree, "log", "-1", "--format=%H%n%B", named)
	if err != nil {
		return "", err
	}
	commit, body, _ := strings.Cut(string(message), "\n")
	for row := range strings.SplitSeq(reference, ",") {
		if !strings.Contains(body, "\n"+plan.LandedRowTrailer+": "+strings.TrimSpace(row)+"\n") {
			return "", fmt.Errorf("loop: the gate reported success but %.12s is not the landing of %s", commit, row)
		}
	}
	if _, err := gitauthority.Query(context.Background(), loopWorktree, "merge-base", "--is-ancestor", commit, "HEAD"); err != nil {
		return "", fmt.Errorf("loop: the gate reported %.12s as the landing of %s, and HEAD does not hold it", commit, reference)
	}
	return commit, nil
}

// Cleared clears, on the real machine, the places a commit clears of what
// nothing holds (clearFinished). A machine that cannot be read clears nothing.
func (w *execRowWorld) Cleared(reference string) string {
	machine, err := realMachine("")
	if err != nil {
		return "nothing was cleared: " + err.Error()
	}
	return clearFinished(machine, landedRows(reference))
}

// standardInput names the landing's own input as the message file.
const standardInput = "-"

// landCommand runs the row sequence and prints its typed outcome; anything
// short of a confirmed landing is a failure exit. A refusal is classed and
// recorded in the checkout's process state, so the outcome says whether the
// change or a cause outside it refused the landing, and how often that
// class of refusal recurred.
func landCommand(reference, messageFile string, input io.Reader, output io.Writer) error {
	if strings.Count(reference, "/") != strings.Count(reference, ",")+1 {
		return fmt.Errorf("usage: loop -land <item>/<step>[,<item>/<step>...] [-message-file <file, or - for standard input>]")
	}
	var message []byte
	if messageFile == standardInput {
		var err error
		if message, err = io.ReadAll(input); err != nil {
			return err
		}
	}
	outcome, err := landRow(&execRowWorld{message: message}, reference, messageFile)
	if err == nil && outcome.Outcome == outcomeReady {
		// The commit landed whatever the lanes do: a runner that does not start is said, and fails the command.
		outcome.Lanes, err = lanesOnClose(loopWorktree, reference, outcome.Commit, startLanesRunner)
		if err != nil {
			outcome.Lanes = "the cycle's lanes did not start: " + err.Error()
		}
	}
	if outcome.Detail != "" {
		refusal, recordErr := landing.RecordRefusal(loopWorktree, landing.Classify(reference, outcome.Detail))
		outcome.Refusal, err = &refusal, errors.Join(err, recordErr)
		if refusal.External != "" {
			outcome.Outcome = outcomeExternal
		}
	}
	if encodeErr := json.NewEncoder(output).Encode(outcome); err == nil {
		err = encodeErr
	}
	if err == nil && outcome.Outcome != outcomeReady {
		err = fmt.Errorf("%s %s at %s", reference, outcome.Outcome, outcome.Phase)
	}
	return err
}

// withheldFromLandings are the settings a landing's tools never inherit from the session that starts them: each lane
// opts into the card (OVERGO_CUDA_TEST) and the live store (testskip.StoreAcceptanceEnv) itself, and a CUDA test's
// measured child (cudatest) is marked by its parent alone, so a session that holds one of them would run every test of
// the gate under it.
var withheldFromLandings = []string{"OVERGO_CUDA_TEST", testskip.StoreAcceptanceEnv, "OVERGO_CUDA_MEASUREMENT_TEST"}

// configuredEnvironment is what the loop's configuration sets in its environment (config.Environment, the role and the
// data roots), none where the checkout keeps no configuration. One it cannot reach, read or decode is refused.
func configuredEnvironment() (map[string]string, error) {
	path, err := processlock.StateFile(loopWorktree, processcontrol.LoopConfigFile, processcontrol.LegacyLoopConfigFile)
	if err != nil {
		return nil, fmt.Errorf("loop: the configuration: %w", err)
	}
	var loaded config
	if err := jsonfile.DecodeStrict(path, &loaded); errors.Is(err, os.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, fmt.Errorf("loop: the configuration %s: %w", path, err)
	}
	return loaded.Environment, nil
}

// landingEnvironment is the environment a landing's tools run in, from environ, the session's: each setting configured
// names that environ leaves unset or empty is set to its configured value, so a landing needs nothing typed before it
// (fail-early-runs/land-environment: guidance-ops/moss-presets, landed with no data root, was refused after its full
// lanes for want of cuDNN), and the settings withheldFromLandings names are dropped from both, the configuration's
// among them (SQA-380: a configuration naming one put it back into every check).
func landingEnvironment(environ []string, configured map[string]string) []string {
	withheld := func(name string) bool {
		return slices.ContainsFunc(withheldFromLandings, func(held string) bool { return strings.EqualFold(held, name) })
	}
	set := map[string]bool{}
	var kept []string
	for _, entry := range environ {
		name, value, _ := strings.Cut(entry, "=")
		if withheld(name) {
			continue
		}
		if value != "" {
			set[strings.ToUpper(name)] = true
		}
		kept = append(kept, entry)
	}
	for _, name := range slices.Sorted(maps.Keys(configured)) {
		if !set[strings.ToUpper(name)] && !withheld(name) {
			kept = append(kept, name+"="+configured[name])
		}
	}
	return kept
}
