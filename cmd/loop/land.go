package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/landing"
	"overgo/internal/plan"
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
}

// rowWorld is what the row sequence drives. A step that refuses returns its
// reason as text; an error is the step being unable to run at all. Cleared
// removes what nothing holds and says what it did: it has no failure to give.
type rowWorld interface {
	Gate(reference, messageFile string) (failure string, err error)
	Landed(reference string) (commit string, err error)
	Cleared() string
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
	outcome.Commit, outcome.Outcome, outcome.Cleared = commit, outcomeReady, world.Cleared()
	return outcome, nil
}

// execRowWorld runs the sequence against the repository's real owners. Its
// subprocesses keep the caller's execution mode. The message is the gate's
// standard input when the worker gave it on the landing's.
type execRowWorld struct {
	message []byte
	// reported is the commit the gate said the row landed as.
	reported string
}

// landedAs opens the commit in the gate's line that says a row landed.
const landedAs = " landed as "

// tool runs one repository tool, echoing its standard output -- a tool's
// verdict; its progress is standard error -- and returns a failed run's
// refusal: the final error the tool wrote to its error file, or the tail of
// its standard error when it wrote none, as a build that never ran it.
func (w *execRowWorld) tool(echo io.Writer, arguments ...string) (string, error) {
	errorFile := filepath.Join(os.TempDir(), fmt.Sprintf("overgo-loop-%d.error", os.Getpid()))
	_ = os.Remove(errorFile)
	// A failed run's file is read below and is then nobody's: it does not outlive the run.
	defer os.Remove(errorFile)
	progress := clioptions.NewTailWriter(clioptions.DiagnosticTailBytes)
	err := runToolEnv(append(os.Environ(), clioptions.ErrorFileEnvironment+"="+errorFile), bytes.NewReader(w.message), echo, progress, arguments...)
	if err == nil {
		return "", nil
	}
	final, readErr := os.ReadFile(errorFile)
	if readErr != nil {
		final = []byte(progress.Tail())
	}
	return cmp.Or(strings.TrimSpace(string(final)), err.Error()), err
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
	refusal, err := w.tool(io.Discard, "go", "build", "-o", gate, "./cmd/gate")
	if err == nil {
		var said bytes.Buffer
		refusal, _ = w.tool(io.MultiWriter(os.Stderr, &said), gate, "-plan", reference, "-message-file", messageFile)
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
func (w *execRowWorld) Cleared() string {
	machine, err := realMachine("")
	if err != nil {
		return "nothing was cleared: " + err.Error()
	}
	return clearFinished(machine)
}

// standardInput names the landing's own input as the message file.
const standardInput = "-"

// landCommand runs the row sequence and prints its typed outcome; anything
// short of a confirmed landing is a failure exit. A refusal is classed and
// recorded in the checkout's process state, so the outcome says whether the
// change or a cause outside it refused the landing, and how often that
// class of refusal recurred.
func landCommand(reference, messageFile string, input io.Reader, output io.Writer) error {
	if strings.Count(reference, "/") != strings.Count(reference, ",")+1 || strings.TrimSpace(messageFile) == "" {
		return fmt.Errorf("usage: loop -land <item>/<step>[,<item>/<step>...] -message-file <file, or - for standard input>")
	}
	var message []byte
	if messageFile == standardInput {
		var err error
		if message, err = io.ReadAll(input); err != nil {
			return err
		}
	}
	outcome, err := landRow(&execRowWorld{message: message}, reference, messageFile)
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
