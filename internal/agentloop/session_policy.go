package agentloop

import "overgo/internal/executionfailure"

// SessionHealth derives the failure policy's session signals from the
// coordinator's own bounded step budget and the caller's recovery
// evidence: a session at the step budget is context-full, and a
// session whose recorded history could not be trusted during recovery
// is poisoned.
func (c *Coordinator) SessionHealth(session *Session, historyTrusted bool) executionfailure.SessionHealth {
	if c == nil || session == nil {
		return executionfailure.SessionHealth{HistoryPoisoned: true}
	}
	return executionfailure.SessionHealth{
		ContextFull:     session.Steps >= c.maxSteps,
		HistoryPoisoned: !historyTrusted,
	}
}

// SessionDisposition derives the canonical post-failure decision for
// one agent session: typed failure evidence and the bounded attempt
// history feed the deterministic matrix, and the returned disposition
// carries the rule and situation so the decision publishes as evidence
// instead of being inferred later from logs.
func (c *Coordinator) SessionDisposition(
	session *Session, classification executionfailure.Classification,
	interrupted bool, attempts, maxAttempts uint32, historyTrusted bool,
) executionfailure.Disposition {
	return executionfailure.Decide(executionfailure.Situation{
		Cause:       classification.Cause,
		Interrupted: interrupted,
		Attempts:    attempts,
		MaxAttempts: maxAttempts,
		Health:      c.SessionHealth(session, historyTrusted),
	})
}
