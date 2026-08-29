package executionfailure

import "slices"

// Decision is one canonical post-failure disposition.
type Decision string

const (
	// DecisionRetry re-executes in the same session and route.
	DecisionRetry Decision = "retry"
	// DecisionResume continues the interrupted session without re-running.
	DecisionResume Decision = "resume"
	// DecisionFreshSession re-attempts in a new session, leaving the old
	// session's context behind.
	DecisionFreshSession Decision = "fresh-session"
	// DecisionRetireSession stops using the session with no automatic
	// re-attempt; an authority decides what happens next.
	DecisionRetireSession Decision = "retire-session"
	// DecisionRefuse never re-executes automatically.
	DecisionRefuse Decision = "refuse"
)

// CanonicalDecisions is the closed decision vocabulary in declaration
// order.
var CanonicalDecisions = []Decision{
	DecisionRetry, DecisionResume, DecisionFreshSession,
	DecisionRetireSession, DecisionRefuse,
}

// ValidDecision reports membership in the canonical decision vocabulary.
func ValidDecision(decision Decision) bool {
	return slices.Contains(CanonicalDecisions, decision)
}

// SessionHealth carries the session signals the policy reads.
type SessionHealth struct {
	// ContextFull reports a session whose context budget is spent.
	ContextFull bool `json:"context_full,omitzero"`
	// HistoryPoisoned reports a session whose recorded history can no
	// longer be trusted, such as after a failed checkpoint restore.
	HistoryPoisoned bool `json:"history_poisoned,omitzero"`
}

// Situation is everything one decision reads: the normalized cause,
// whether the execution was interrupted rather than failed on its own,
// the bounded attempt history, the explicit budget, and session health.
type Situation struct {
	Cause Cause `json:"cause"`
	// Interrupted reports a supervised execution stopped by request.
	Interrupted bool `json:"interrupted,omitzero"`
	// Attempts counts executions already run, this one included.
	Attempts uint32 `json:"attempts"`
	// MaxAttempts is the explicit automatic re-execution budget.
	MaxAttempts uint32        `json:"max_attempts"`
	Health      SessionHealth `json:"health"`
}

// Disposition is the published decision evidence: the decision, the
// matrix rule that produced it, and the exact situation it read.
type Disposition struct {
	Decision  Decision  `json:"decision"`
	Rule      string    `json:"rule"`
	Situation Situation `json:"situation"`
}

// transientCauses may retry within the explicit budget: the failure is
// plausibly environmental, and unknown evidence is treated as transient
// so the budget, not optimism, bounds re-execution.
var transientCauses = []Cause{
	CauseNetwork, CauseTimeout, CauseCapacity, CauseQuota,
	CauseModelAvailability, CauseUnknown,
}

// Rule names for the decision matrix below, published with every
// disposition so the decision is evidence rather than log inference.
const (
	rulePoisonedHistory       = "poisoned-history"
	ruleInterruptedHealthy    = "interrupted-healthy"
	ruleContextRollover       = "context-rollover"
	ruleContextBudgetSpent    = "context-budget-spent"
	ruleTransientWithinBudget = "transient-within-budget"
	ruleBudgetExhausted       = "budget-exhausted"
	ruleDeterministicLocal    = "deterministic-local"
)

// Decide is the deterministic decision matrix, evaluated top to
// bottom: poisoned history retires the session outright; a healthy
// interrupted session resumes; a full context rolls to a fresh session
// while budget remains and retires after; transient causes retry
// within the budget and refuse beyond it; deterministic local causes
// never retry.
func Decide(situation Situation) Disposition {
	derived := func(decision Decision, rule string) Disposition {
		return Disposition{Decision: decision, Rule: rule, Situation: situation}
	}
	withinBudget := situation.Attempts < situation.MaxAttempts
	switch {
	case situation.Health.HistoryPoisoned:
		return derived(DecisionRetireSession, rulePoisonedHistory)
	case situation.Interrupted && !situation.Health.ContextFull:
		return derived(DecisionResume, ruleInterruptedHealthy)
	case situation.Health.ContextFull || situation.Cause == CauseContextExhaustion:
		if withinBudget {
			return derived(DecisionFreshSession, ruleContextRollover)
		}
		return derived(DecisionRetireSession, ruleContextBudgetSpent)
	case slices.Contains(transientCauses, situation.Cause):
		if withinBudget {
			return derived(DecisionRetry, ruleTransientWithinBudget)
		}
		return derived(DecisionRefuse, ruleBudgetExhausted)
	default:
		return derived(DecisionRefuse, ruleDeterministicLocal)
	}
}
