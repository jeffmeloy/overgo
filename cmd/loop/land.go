package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
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
// an interactive row.
type execRowWorld struct{ store *overgodb.Store }

func (w execRowWorld) tool(arguments ...string) (string, error) {
	return runToolEnv(os.Environ(), os.Stderr, arguments...)
}

// refusalOf keeps the lines of a failed tool run that say why.
func refusalOf(output string) string {
	var kept []string
	for line := range strings.SplitSeq(output, "\n") {
		if strings.Contains(line, "FAIL") || strings.Contains(line, "blocker:") || strings.Contains(line, "refused") ||
			strings.HasPrefix(line, "plan:") || strings.HasPrefix(line, "gate: gate:") || strings.HasPrefix(line, "preflight: ") && strings.Contains(line, "finding") {
			kept = append(kept, strings.TrimSpace(line))
		}
	}
	if len(kept) == 0 {
		return strings.TrimSpace(output)
	}
	return strings.Join(kept, "\n")
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

// landCommand runs the row sequence and prints its typed outcome; anything
// short of a validated landing is a failure exit.
func landCommand(reference, messageFile string, output io.Writer) error {
	if strings.Count(reference, "/") != 1 || strings.TrimSpace(messageFile) == "" {
		return fmt.Errorf("usage: loop -land <item>/<step> -message-file <file>")
	}
	store, err := overgodb.OpenReadOnly(gitauthority.CanonicalOvergoDBDirectory)
	if err != nil {
		return err
	}
	defer store.Close()
	outcome, err := landRow(execRowWorld{store: store}, reference, messageFile)
	if encodeErr := json.NewEncoder(output).Encode(outcome); err == nil {
		err = encodeErr
	}
	if err == nil && outcome.Outcome != outcomeReady && outcome.Outcome != outcomeReviewPending {
		err = fmt.Errorf("%s %s at %s", reference, outcome.Outcome, outcome.Phase)
	}
	return err
}
