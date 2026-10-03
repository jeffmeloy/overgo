// Package loop is the harness-agnostic continuous-work driver: it owns the
// plan -> implement -> verify -> commit cycle with NO concept of a turn or a
// stopping model. The plan, verifier, and gate are deterministic Go owned by
// this repository; the only model-dependent step is "implement", which runs as
// a pluggable worker subprocess. Worker exit -- clean, early, crashed, or
// bored -- is just an event: the driver re-reads the plan and either advances,
// retries the same step with the failure tail as feedback, or parks loudly
// after the attempt budget. Continuity is a for-loop in a process that cannot
// be talked out of anything, so it works regardless of harness or model. When
// the plan empties, the scorecard measures what the work changed, a holding
// held-out quality promotes it, and the gaps become the next rows.
package loop

import (
	"errors"
	"fmt"
	"strings"

	"overgo/internal/scorecard"
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

// ProgressWorld recognizes accepted prerequisite work while the dispatched
// step remains open. The checkpoint precedes the worker; neither a changed
// revision nor a green verifier alone establishes accepted progress.
type ProgressWorld interface {
	ProgressCheckpoint() (string, error)
	AcceptedProgress(step Step, checkpoint string) (bool, error)
}

// Config bounds the driver. Every limit is mechanical: no prose persuasion.
type Config struct {
	// MaxAttemptsPerStep parks the step after this many worker exits
	// without plan advancement or accepted prerequisite progress. Zero
	// refuses to run: unbounded retries on a stuck step are forbidden.
	MaxAttemptsPerStep int `json:"max_attempts_per_step"`
	// MaxInvocations bounds total worker launches for one Run call.
	MaxInvocations int `json:"max_invocations"`
}

// ObligationWorld is the optional world extension that replays durable
// loop obligations. The driver calls ReplayObligations at the top of
// every supervision round, before reading the plan and whether or not
// anything else fires, so a follow-up admitted before a crash is
// repaired from committed facts instead of remembered state.
type ObligationWorld interface {
	// ReplayObligations completes every obligation whose predicate now
	// holds and returns how many follow-ups remain due.
	ReplayObligations() (due uint64, err error)
}

// ImprovementWorld is the optional world extension that closes the
// improvement loop each time the plan empties: the scorecard run is the
// promotion evidence, the promoted branch moves only when the run's
// held-out quality holds against its baseline, and the run's gaps return as
// plan rows, the next work.
type ImprovementWorld interface {
	// Measure runs the scorecard at the main checkout's HEAD, in a checkout
	// apart from it so rows land while it runs, and returns the record it
	// published and the baseline that record is judged against, nil when
	// there is none. A failed validation returns its error beside the
	// published record; a run that published nothing returns no record.
	Measure() (published, baseline *scorecard.Record, err error)
	// Promote moves the promoted branch forward to the measured commit. It
	// never pushes.
	Promote(commit string) error
	// Propose writes the gaps the plan does not hold as rows and reports how
	// many it added.
	Propose(published scorecard.Record, gaps []scorecard.Gap) (added int, err error)
	// Fill runs the validations promotion does not compare at the commit
	// the scorecard measured, after the cycle's decision, so their evidence
	// flows without delaying or withholding a promotion. It reports its own
	// failure.
	Fill(commit string)
}

// Outcome reports why Run returned.
type Outcome struct {
	Reason      string
	Invocations int
	Parked      []string
	// ObligationsDue is the last replay's count of follow-ups still owed.
	ObligationsDue uint64
	// Promoted is the commit the last holding scorecard promoted, if any.
	Promoted string
	// Withheld is why the last scorecard promoted nothing, if it did not.
	Withheld string
}

const (
	ReasonPlanComplete = "plan-complete"
	// ReasonWorkWaiting preserves eligible work while another worker owns its dispatch.
	ReasonWorkWaiting = "work-ownership-waiting"
	ReasonPaused      = "paused"
	ReasonBudget      = "invocation-budget-exhausted"
	ReasonParked      = "step-parked"
)

// ErrWorkWaiting marks expected ownership contention, not a failed task.
var ErrWorkWaiting = errors.New("loop: waiting for work ownership")

// Run drives the cycle until a terminal condition. It returns an error only
// when the world itself fails (plan unreadable, worker unlaunchable); a parked
// step is a terminal outcome, not an error, so the operator decides.
func Run(world World, config Config) (Outcome, error) {
	if config.MaxAttemptsPerStep <= 0 || config.MaxInvocations <= 0 {
		return Outcome{}, errors.New("loop: attempt and invocation budgets are required")
	}
	outcome := Outcome{}
	attempts := 0
	progress, observesProgress := world.(ProgressWorld)
	obligations, replays := world.(ObligationWorld)
	var lastKey, feedback string
	for {
		if world.Paused() {
			outcome.Reason = ReasonPaused
			return outcome, nil
		}
		if replays {
			due, err := obligations.ReplayObligations()
			if err != nil {
				return outcome, fmt.Errorf("loop: replay obligations: %w", err)
			}
			outcome.ObligationsDue = due
		}
		step, open, err := world.Current()
		if world.Paused() {
			outcome.Reason = ReasonPaused
			return outcome, nil
		}
		if errors.Is(err, ErrWorkWaiting) {
			outcome.Reason = ReasonWorkWaiting
			return outcome, nil
		}
		if err != nil {
			return outcome, fmt.Errorf("loop: read plan: %w", err)
		}
		if !open {
			if added, err := improve(world, &outcome); err != nil || added == 0 {
				outcome.Reason = ReasonPlanComplete
				return outcome, err
			}
			continue
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
		var checkpoint string
		if observesProgress {
			checkpoint, err = progress.ProgressCheckpoint()
			if err != nil {
				return outcome, fmt.Errorf("loop: checkpoint progress: %w", err)
			}
		}
		tail, err := world.RunWorker(step, prompt, feedback)
		outcome.Invocations++
		attempts++
		if err != nil {
			// A worker that cannot even launch is a world failure; a worker
			// that exited nonzero is ordinary -- the plan decides below.
			if _, ok := errors.AsType[*LaunchError](err); ok {
				return outcome, fmt.Errorf("loop: worker launch: %w", err)
			}
		}
		if world.Paused() {
			outcome.Reason = ReasonPaused
			return outcome, nil
		}
		after, open, err := world.Current()
		if err != nil {
			return outcome, fmt.Errorf("loop: re-read plan: %w", err)
		}
		if !open || after.Key() != step.Key() {
			// The worker advanced the step through the gate.
			continue
		}
		if observesProgress {
			accepted, err := progress.AcceptedProgress(step, checkpoint)
			if err != nil {
				return outcome, fmt.Errorf("loop: accepted progress: %w", err)
			}
			if accepted {
				// Only the consecutive nonprogress streak resets. This worker
				// still consumed an invocation, and the parent is still open.
				attempts = 0
				feedback = "Accepted prerequisite work; remaining acceptance is still open: " + step.Key()
				continue
			}
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

// improve runs one improvement cycle when the plan empties. A world that
// does not measure adds nothing.
func improve(world World, outcome *Outcome) (int, error) {
	improver, measures := world.(ImprovementWorld)
	if !measures {
		return 0, nil
	}
	return Improve(improver, outcome)
}

// Improve runs one improvement cycle: it measures the scorecard, promotes
// the measured commit only when the record is not withheld (a baseline
// measured on the promoted branch, every held-out quality measured, none
// fallen), proposes the gaps as rows, then fills the validations promotion
// does not compare at the measured commit. The measurement and the fill
// record their claims in a checkout of that commit apart from the main one,
// so rows land while they run and the proposals do not wait on the fill:
// the rows reach the plan before the fill's hours start, and a fill that is
// stopped loses none of them. A failed validation leaves stale cells, so it
// promotes nothing; its record still names the gaps. The outcome says why a
// commit was withheld, a failed validation by the failure Measure returned.
// The driver runs it when the plan empties, and cmd/loop -improve on demand.
func Improve(improver ImprovementWorld, outcome *Outcome) (int, error) {
	published, baseline, err := improver.Measure()
	if published == nil {
		return 0, errors.Join(errors.New("loop: no scorecard was published"), err)
	}
	outcome.Withheld = published.Withheld(baseline)
	if err != nil {
		outcome.Withheld = "a validation failed, so the scorecard's cells are stale: " + err.Error()
	}
	if outcome.Withheld == "" {
		if err := improver.Promote(published.Commit); err != nil {
			return 0, fmt.Errorf("loop: promote: %w", err)
		}
		outcome.Promoted = published.Commit
	}
	added, proposeErr := improver.Propose(*published, published.Gaps(baseline))
	improver.Fill(published.Commit)
	return added, errors.Join(err, proposeErr)
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
