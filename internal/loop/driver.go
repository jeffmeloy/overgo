// Package loop is the harness-agnostic continuous-work driver: it owns the
// plan -> implement -> verify -> commit cycle with NO concept of a turn or a
// stopping model. The plan, verifier, and gate are deterministic Go owned by
// this repository; the only model-dependent step is "implement", which runs as
// a pluggable worker subprocess. Worker exit -- clean, early, crashed, or
// bored -- is just an event: the driver re-reads the plan and either advances,
// retries the same step with the failure tail as feedback, or parks loudly
// after the attempt budget. Continuity is a for-loop in a process that cannot
// be talked out of anything, so it works regardless of harness or model.
package loop

import (
	"errors"
	"fmt"
	"strings"
)

// Step identifies one dispatched unit of plan work.
type Step struct {
	Item string
	ID   string
}

func (s Step) Key() string { return s.Item + "/" + s.ID }

// World is what the driver observes and drives. Implementations shell out to
// cmd/plan and the configured worker; tests supply fakes. The driver never
// commits or advances anything itself -- the worker commits through the gate,
// exactly as an interactive session would, and the gate remains the sole
// commit path.
type World interface {
	// Current returns the dispatched step, or ok=false when no open step
	// remains (plan complete or blocked-only).
	Current() (Step, bool, error)
	// Prompt renders the dispatch prompt for the worker (plan -prompt).
	Prompt(step Step) (string, error)
	// RunWorker executes one worker invocation against the prompt plus
	// accumulated feedback and blocks until the worker process exits.
	// The returned transcript tail is used only for feedback and logging.
	RunWorker(step Step, prompt, feedback string) (tail string, err error)
	// Verify runs the step's verifier (plan -verify) and returns its
	// failure output when it fails; empty output means the verifier passed.
	Verify(step Step) (failure string, err error)
	// Park records a finding for a step that exhausted its attempts.
	Park(step Step, reason string) error
	// Paused reports the kill-switch marker.
	Paused() bool
}

// Config bounds the driver. Every limit is mechanical: no prose persuasion.
type Config struct {
	// MaxAttemptsPerStep parks the step after this many worker exits
	// without plan advancement. Zero refuses to run: an unbounded retry
	// loop on a stuck step is the failure mode this exists to prevent.
	MaxAttemptsPerStep int `json:"max_attempts_per_step"`
	// MaxInvocations bounds total worker launches for one Run call.
	MaxInvocations int `json:"max_invocations"`
}

// Outcome reports why Run returned.
type Outcome struct {
	Reason      string
	Invocations int
	Parked      []string
}

const (
	ReasonPlanComplete = "plan-complete"
	ReasonPaused       = "paused"
	ReasonBudget       = "invocation-budget-exhausted"
	ReasonParked       = "step-parked"
)

// Run drives the cycle until a terminal condition. It returns an error only
// when the world itself fails (plan unreadable, worker unlaunchable); a parked
// step is a terminal outcome, not an error, so the operator decides.
func Run(world World, config Config) (Outcome, error) {
	if config.MaxAttemptsPerStep <= 0 || config.MaxInvocations <= 0 {
		return Outcome{}, errors.New("loop: attempt and invocation budgets are required")
	}
	outcome := Outcome{}
	attempts := 0
	var lastKey, feedback string
	for {
		if world.Paused() {
			outcome.Reason = ReasonPaused
			return outcome, nil
		}
		step, open, err := world.Current()
		if err != nil {
			return outcome, fmt.Errorf("loop: read plan: %w", err)
		}
		if !open {
			outcome.Reason = ReasonPlanComplete
			return outcome, nil
		}
		if step.Key() != lastKey {
			lastKey, attempts, feedback = step.Key(), 0, ""
		}
		if attempts >= config.MaxAttemptsPerStep {
			reason := fmt.Sprintf("loop: %s made no plan progress in %d worker attempts; last verify failure: %s",
				step.Key(), attempts, truncate(feedback, 500))
			if err := world.Park(step, reason); err != nil {
				return outcome, fmt.Errorf("loop: park %s: %w", step.Key(), err)
			}
			outcome.Parked = append(outcome.Parked, step.Key())
			outcome.Reason = ReasonParked
			return outcome, nil
		}
		if outcome.Invocations >= config.MaxInvocations {
			outcome.Reason = ReasonBudget
			return outcome, nil
		}
		prompt, err := world.Prompt(step)
		if err != nil {
			return outcome, fmt.Errorf("loop: render prompt: %w", err)
		}
		tail, err := world.RunWorker(step, prompt, feedback)
		outcome.Invocations++
		attempts++
		if err != nil {
			// A worker that cannot even launch is a world failure; a worker
			// that exited nonzero is ordinary -- the plan decides below.
			var launch *LaunchError
			if errors.As(err, &launch) {
				return outcome, fmt.Errorf("loop: worker launch: %w", err)
			}
		}
		after, open, err := world.Current()
		if err != nil {
			return outcome, fmt.Errorf("loop: re-read plan: %w", err)
		}
		if !open || after.Key() != step.Key() {
			continue // the worker advanced the step through the gate
		}
		failure, err := world.Verify(step)
		if err != nil {
			return outcome, fmt.Errorf("loop: verify: %w", err)
		}
		if failure == "" {
			// Verifier green but not committed: the next attempt's feedback
			// says exactly that, so the worker finishes through the gate.
			feedback = "verifier already passes; commit through the gate and advance: " + step.Key()
			continue
		}
		feedback = "attempt " + fmt.Sprint(attempts) + " verify failure:\n" + truncate(failure, 2000) +
			"\nworker tail:\n" + truncate(tail, 1000)
	}
}

// LaunchError marks a worker that could not start at all, as distinct from a
// worker that ran and exited nonzero.
type LaunchError struct{ Err error }

func (e *LaunchError) Error() string { return e.Err.Error() }
func (e *LaunchError) Unwrap() error { return e.Err }

func truncate(value string, limit int) string {
	value = strings.TrimSpace(value)
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "...[truncated]"
}
