package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// Landing a row is a fixed sequence around the one thing a worker authors:
// claim the row, preflight it, gate it, confirm the landing from git, read
// the deferred lanes' verdict, and read the landing's review candidates. The
// sequence is code so that no step can be skipped or reordered by memory --
// a gate launched without a claim, a lane verdict never read -- and every
// run ends in one typed outcome.

// Row sequence phases, in order.
const (
	phaseClaim     = "claim"
	phasePreflight = "preflight"
	phaseGate      = "gate"
	phaseLanes     = "lanes"
	phaseReview    = "review"
)

// Row sequence outcomes.
const (
	// outcomeRefused stops before or at the gate: nothing landed.
	outcomeRefused = "refused"
	// outcomeLanesFailed landed a commit whose deferred validation failed.
	outcomeLanesFailed = "lanes-failed"
	// outcomeReviewPending landed and validated; dispatch waits on answers.
	outcomeReviewPending = "review-pending"
	// outcomeReady landed, validated, and nothing is left to answer.
	outcomeReady = "ready"
)

// landOutcome is the typed result of one row sequence.
type landOutcome struct {
	Reference string   `json:"reference"`
	Phase     string   `json:"phase"`
	Outcome   string   `json:"outcome"`
	Commit    string   `json:"commit,omitzero"`
	Detail    string   `json:"detail,omitzero"`
	Pending   []string `json:"pending,omitempty"`
}

// rowWorld is what the row sequence drives. A step that refuses returns its
// reason as text; an error is the step being unable to run at all.
type rowWorld interface {
	Claim(reference string) (refusal string, err error)
	Preflight(reference string) (findings string, err error)
	Gate(reference, messageFile string) (failure string, err error)
	Landed(reference string) (commit string, err error)
	AwaitLanes() (failure string, err error)
	PendingReview() ([]string, error)
}

// landRow runs the sequence and stops at the first phase that does not pass.
func landRow(world rowWorld, reference, messageFile string) (landOutcome, error) {
	outcome := landOutcome{Reference: reference, Outcome: outcomeRefused}
	for _, step := range []struct {
		phase string
		run   func() (string, error)
	}{
		{phaseClaim, func() (string, error) { return world.Claim(reference) }},
		{phasePreflight, func() (string, error) { return world.Preflight(reference) }},
		{phaseGate, func() (string, error) { return world.Gate(reference, messageFile) }},
	} {
		outcome.Phase = step.phase
		refusal, err := step.run()
		if err != nil || refusal != "" {
			outcome.Detail = refusal
			return outcome, err
		}
	}
	commit, err := world.Landed(reference)
	if err != nil {
		return outcome, err
	}
	outcome.Commit, outcome.Phase = commit, phaseLanes
	failure, err := world.AwaitLanes()
	if err != nil || failure != "" {
		outcome.Outcome, outcome.Detail = outcomeLanesFailed, failure
		return outcome, err
	}
	outcome.Phase = phaseReview
	if outcome.Pending, err = world.PendingReview(); err != nil {
		return outcome, err
	}
	outcome.Outcome = outcomeReady
	if len(outcome.Pending) != 0 {
		outcome.Outcome = outcomeReviewPending
	}
	return outcome, nil
}

// execRowWorld runs the sequence against the repository's real owners. Its
// subprocesses keep the caller's execution mode: an interactive worker lands
// an interactive row. The message is the gate's standard input when the worker
// gave it on the landing's; a tool that reads none ignores it.
type execRowWorld struct {
	store   *overgodb.Store
	message []byte
}

func (w execRowWorld) tool(arguments ...string) (string, error) {
	return runToolEnv(os.Environ(), bytes.NewReader(w.message), &verdictLines{out: os.Stderr}, arguments...)
}

// verdictLines passes on the lines of a tool's output that decide or explain
// the outcome: a refusal, a failing test with the detail indented under it,
// the gate's verdict and what it measured. Everything else a landing prints
// -- selection lists, a line per package -- is a record in the store already,
// so a worker has no transcript worth saving to a file.
type verdictLines struct {
	out     io.Writer
	partial []byte
	failing bool
}

var verdictLine = regexp.MustCompile(`FAIL|result=fail|blocker:|refused|^(?:plan:|gate: gate:|GATE |advisory: (?:class|delta|debt|warning):|preflight: .*finding)`)

// Write echoes the whole lines completed by data that the verdict keeps.
func (v *verdictLines) Write(data []byte) (int, error) {
	v.partial = append(v.partial, data...)
	for {
		line, rest, whole := bytes.Cut(v.partial, []byte{'\n'})
		if !whole {
			return len(data), nil
		}
		v.partial = rest
		if v.keeps(string(line)) {
			if _, err := fmt.Fprintf(v.out, "%s\n", line); err != nil {
				return len(data), err
			}
		}
	}
}

func (v *verdictLines) keeps(line string) bool {
	detail := v.failing && strings.TrimLeft(line, " \t") != line
	v.failing = detail || strings.Contains(line, "--- FAIL")
	return detail || verdictLine.MatchString(line)
}

// refusalOf keeps the lines of a failed tool run that say why.
func refusalOf(output string) string {
	var kept strings.Builder
	_, _ = (&verdictLines{out: &kept}).Write([]byte(output + "\n"))
	return strings.TrimSpace(cmp.Or(kept.String(), output))
}

// Claim dispatches the row to this worker; the holder re-claims idempotently.
func (w execRowWorld) Claim(reference string) (string, error) {
	out, err := w.tool("go", "run", "./cmd/plan", "-prompt", reference)
	if err != nil {
		return refusalOf(out), nil
	}
	return "", nil
}

// Preflight diagnoses the dirty tree without admission or acceptance credit.
func (w execRowWorld) Preflight(reference string) (string, error) {
	out, err := w.tool("go", "run", "./cmd/gate", "-preflight", "-plan", reference)
	if err != nil {
		return refusalOf(out), nil
	}
	return "", nil
}

// Gate commits the row over the ship set the gate derives.
func (w execRowWorld) Gate(reference, messageFile string) (string, error) {
	out, err := w.tool("go", "run", "./cmd/gate", "-plan", reference, "-message-file", messageFile)
	if err != nil {
		return refusalOf(out), nil
	}
	return "", nil
}

// Landed confirms from git that HEAD is the completion of reference.
func (w execRowWorld) Landed(reference string) (string, error) {
	item, step, _ := strings.Cut(reference, "/")
	message, err := gitauthority.Query(context.Background(), loopWorktree, "log", "-1", "--format=%H%n%B")
	if err != nil {
		return "", err
	}
	commit, body, _ := strings.Cut(string(message), "\n")
	if !strings.Contains(body, "\nOvergo-Plan-Item: "+item+"\n") || !strings.Contains(body, "\nOvergo-Plan-Step: "+step+"\n") {
		return "", fmt.Errorf("loop: the gate reported success but HEAD %.12s is not the completion of %s", commit, reference)
	}
	return commit, nil
}

// AwaitLanes waits on the lane obligation the way the driver does.
func (w execRowWorld) AwaitLanes() (string, error) {
	driver := &execWorld{stopStore: w.store}
	err := driver.awaitValidation()
	return driver.validationDebt, err
}

// PendingReview lists the landing's unanswered optimization candidates.
func (w execRowWorld) PendingReview() ([]string, error) {
	if err := w.store.Refresh(context.Background()); err != nil {
		return nil, err
	}
	document, err := plan.Load("")
	if err != nil {
		return nil, err
	}
	candidates, err := plan.ReviewLandedCompletion(context.Background(), w.store, loopWorktree, "HEAD")
	if err != nil {
		return nil, err
	}
	var pending []string
	for _, candidate := range plan.UnreviewedCandidates(candidates, document) {
		pending = append(pending, candidate.Kind+" "+candidate.Key+": "+candidate.Measure)
	}
	return pending, nil
}

// standardInput names the landing's own input as the message file.
const standardInput = "-"

// landCommand runs the row sequence and prints its typed outcome; anything
// short of a validated landing is a failure exit.
func landCommand(reference, messageFile string, input io.Reader, output io.Writer) error {
	if strings.Count(reference, "/") != 1 || strings.TrimSpace(messageFile) == "" {
		return fmt.Errorf("usage: loop -land <item>/<step> -message-file <file, or - for standard input>")
	}
	store, err := overgodb.OpenReadOnly(gitauthority.CanonicalOvergoDBDirectory)
	if err != nil {
		return err
	}
	defer store.Close()
	var message []byte
	if messageFile == standardInput {
		if message, err = io.ReadAll(input); err != nil {
			return err
		}
	}
	outcome, err := landRow(execRowWorld{store: store, message: message}, reference, messageFile)
	if encodeErr := json.NewEncoder(output).Encode(outcome); err == nil {
		err = encodeErr
	}
	if err == nil && outcome.Outcome != outcomeReady && outcome.Outcome != outcomeReviewPending {
		err = fmt.Errorf("%s %s at %s", reference, outcome.Outcome, outcome.Phase)
	}
	return err
}
