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

// ProposalSource is the optional world extension that feeds the driver
// admitted steering proposals once the plan drains: the loop closure.
// AdmitNext admits the next pending proposal into the plan through
// deterministic admission and returns its plan item id; ok=false means
// the queue is empty. Block marks a parked proposal row blocked so
// dispatch moves past it to the next proposal.
type ProposalSource interface {
	AdmitNext() (item string, ok bool, err error)
	Block(step Step, reason string) error
}

// Config bounds the driver. Every limit is mechanical: no prose persuasion.
type Config struct {
	// MaxAttemptsPerStep parks the step after this many worker exits
	// without plan advancement. Zero refuses to run: an unbounded retry
	// loop on a stuck step is the failure mode this exists to prevent.
	MaxAttemptsPerStep int `json:"max_attempts_per_step"`
	// MaxInvocations bounds total worker launches for one Run call.
	MaxInvocations int            `json:"max_invocations"`
	Closure        *ClosureConfig `json:"closure,omitempty"`
	// SaturationLimit supports the compatibility proposal source when no
	// evidence-bound Closure is configured.
	SaturationLimit int `json:"saturation_limit,omitzero"`
}

// ClosureConfig bounds evidence-gated proposal consumption after the plan
// drains. Every budget must be nonzero; a zero budget stops the loop as
// budget-exhausted rather than running unbounded.
type ClosureConfig struct {
	MaxWallNS      uint64 `json:"max_wall_ns"`
	MaxCostUnits   uint64 `json:"max_cost_units"`
	MaxMutations   uint64 `json:"max_mutations"`
	MaxExperiments uint64 `json:"max_experiments"`
	MaxProposals   uint64 `json:"max_proposals"`
}

// ClosureFacts are the measured world facts the closure stop decision
// consumes; they are observed, never predicted.
type ClosureFacts struct {
	WallNS                 uint64
	CostUnits              uint64
	Mutations              uint64
	Experiments            uint64
	OutstandingObligations uint64
	LeaseConflicts         uint64
	RepeatedDirections     uint64
	MeasuredGain           bool
	OperatorStop           bool
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

// ProposalWorld is an optional extension of the same deterministic driver.
// Admission mutates only the plan; strategy activation remains external and evidence-gated.
type ProposalWorld interface {
	World
	ClosureFacts() (ClosureFacts, error)
	AdmitNextProposal() (bool, error)
}

// Outcome reports why Run returned.
type Outcome struct {
	Reason      string
	Invocations int
	Parked      []string
	// ObligationsDue is the last replay's count of follow-ups still owed.
	ObligationsDue uint64
}

const (
	ReasonPlanComplete = "plan-complete"
	ReasonPaused       = "paused"
	ReasonBudget       = "invocation-budget-exhausted"
	ReasonParked       = "step-parked"
	// ReasonClosureBlocked reports outstanding obligations or lease
	// conflicts blocking further proposal consumption.
	ReasonClosureBlocked = "proposal-closure-blocked"
	// ReasonSaturated reports repeated directions or absent measured gain.
	ReasonSaturated = "proposal-saturated"
	// ReasonOperatorStop reports an explicit operator stop fact.
	ReasonOperatorStop = "operator-stop"
)

// Run drives the cycle until a terminal condition. It returns an error only
// when the world itself fails (plan unreadable, worker unlaunchable); a parked
// step is a terminal outcome, not an error, so the operator decides.
func Run(world World, config Config) (Outcome, error) {
	if config.MaxAttemptsPerStep <= 0 || config.MaxInvocations <= 0 {
		return Outcome{}, errors.New("loop: attempt and invocation budgets are required")
	}
	outcome := Outcome{}
	var proposals uint64
	attempts := 0
	saturation := 0
	proposalItems := map[string]bool{}
	source, feeds := world.(ProposalSource)
	feeds = feeds && config.SaturationLimit > 0
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
		if err != nil {
			return outcome, fmt.Errorf("loop: read plan: %w", err)
		}
		if !open {
			if config.Closure != nil {
				proposalWorld, capable := world.(ProposalWorld)
				if !capable {
					return outcome, errors.New("loop: closure configured without proposal world")
				}
				facts, factErr := proposalWorld.ClosureFacts()
				if factErr != nil {
					return outcome, factErr
				}
				if reason := closureStopReason(*config.Closure, facts, proposals); reason != "" {
					outcome.Reason = reason
					return outcome, nil
				}
				admitted, admitErr := proposalWorld.AdmitNextProposal()
				if admitErr != nil {
					return outcome, admitErr
				}
				if !admitted {
					outcome.Reason = ReasonPlanComplete
					return outcome, nil
				}
				proposals++
				continue
			}
			// The closure: a drained plan consumes the next admitted
			// proposal instead of stopping, until the queue empties or
			// saturation proves further consumption unmeasured.
			if !feeds {
				outcome.Reason = ReasonPlanComplete
				return outcome, nil
			}
			if saturation >= config.SaturationLimit {
				outcome.Reason = ReasonSaturated
				return outcome, nil
			}
			item, pending, err := source.AdmitNext()
			if err != nil {
				return outcome, fmt.Errorf("loop: admit proposal: %w", err)
			}
			if !pending {
				outcome.Reason = ReasonPlanComplete
				return outcome, nil
			}
			proposalItems[item] = true
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
			if feeds && proposalItems[step.Item] {
				// A parked proposal row is no measured improvement: it
				// blocks out of dispatch, counts toward saturation, and
				// the loop moves to the next proposal instead of ending.
				if err := source.Block(step, reason); err != nil {
					return outcome, fmt.Errorf("loop: block %s: %w", step.Key(), err)
				}
				saturation++
				lastKey, attempts, feedback = "", 0, ""
				continue
			}
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
			if _, ok := errors.AsType[*LaunchError](err); ok {
				return outcome, fmt.Errorf("loop: worker launch: %w", err)
			}
		}
		after, open, err := world.Current()
		if err != nil {
			return outcome, fmt.Errorf("loop: re-read plan: %w", err)
		}
		if !open || after.Key() != step.Key() {
			// The worker advanced the step through the gate. An advanced
			// proposal-driven row passed its falsifiable check: measured
			// improvement, saturation streak reset.
			if proposalItems[step.Item] {
				saturation = 0
			}
			continue
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

func closureStopReason(config ClosureConfig, facts ClosureFacts, proposals uint64) string {
	if facts.OperatorStop {
		return ReasonOperatorStop
	}
	if facts.OutstandingObligations != 0 || facts.LeaseConflicts != 0 {
		return ReasonClosureBlocked
	}
	if facts.RepeatedDirections != 0 || !facts.MeasuredGain {
		return ReasonSaturated
	}
	if config.MaxWallNS == 0 || config.MaxCostUnits == 0 || config.MaxMutations == 0 || config.MaxExperiments == 0 || config.MaxProposals == 0 ||
		facts.WallNS >= config.MaxWallNS || facts.CostUnits >= config.MaxCostUnits || facts.Mutations >= config.MaxMutations || facts.Experiments >= config.MaxExperiments || proposals >= config.MaxProposals {
		return ReasonBudget
	}
	return ""
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
