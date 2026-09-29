package runrecord

// triggerAuthority names where a trigger's permission to execute comes
// from. Authority never derives from the causal chain itself.
type triggerAuthority string

const (
	// authorityOperatorApproval requires an explicit operator decision.
	authorityOperatorApproval triggerAuthority = "operator-approval"
	// AuthorityDeclaredPolicy executes under a committed policy document.
	AuthorityDeclaredPolicy triggerAuthority = "declared-policy"
	// AuthorityInherited executes under the admitting execution's own
	// authority, which must already exist.
	AuthorityInherited triggerAuthority = "inherited"
)

// triggerBudget names which budget admits a trigger's execution.
type triggerBudget string

const (
	// budgetInteractive charges the interactive session budget.
	budgetInteractive triggerBudget = "interactive"
	// budgetDeclared charges a budget a committed policy declares.
	budgetDeclared triggerBudget = "declared"
	// budgetAttemptBounded charges the bounded attempt budget.
	budgetAttemptBounded triggerBudget = "attempt-bounded"
	// BudgetInherited charges the admitting execution's budget.
	BudgetInherited triggerBudget = "inherited"
)

// triggerIdempotency names how duplicate firings are prevented.
type triggerIdempotency string

const (
	// idempotencyNone admits every firing; only interactive triggers may
	// declare it.
	idempotencyNone triggerIdempotency = "none"
	// IdempotencyKeyed dedupes firings by an explicit key.
	IdempotencyKeyed triggerIdempotency = "keyed"
	// IdempotencyReplaySafe converges under repetition by construction.
	IdempotencyReplaySafe triggerIdempotency = "replay-safe"
)

// triggerContract is one trigger kind's complete declared behavior:
// how it participates in causality, whose authority admits it, which
// budget it charges, how duplicates are prevented, and the evidence
// record that must exist when it fires.
type triggerContract struct {
	Trigger CausalTrigger
	// Originates reports whether the trigger mints a causal root.
	Originates  bool
	Authority   triggerAuthority
	Budget      triggerBudget
	Idempotency triggerIdempotency
	// Evidence names the record kind that must exist for the firing.
	Evidence string
}

// triggerRegistry declares every canonical trigger's contract, in
// CausalTrigger declaration order. There is no undocumented enqueue
// path: a trigger added to the causal vocabulary without a complete
// contract here fails the registry's test.
var triggerRegistry = []triggerContract{
	{Trigger: TriggerManual, Originates: true, Authority: authorityOperatorApproval,
		Budget: budgetInteractive, Idempotency: idempotencyNone, Evidence: "operator-decision"},
	{Trigger: triggerSchedule, Originates: true, Authority: AuthorityDeclaredPolicy,
		Budget: budgetDeclared, Idempotency: IdempotencyKeyed, Evidence: "schedule-slot"},
	{Trigger: TriggerWebhook, Originates: true, Authority: AuthorityDeclaredPolicy,
		Budget: budgetDeclared, Idempotency: IdempotencyKeyed, Evidence: "ingress-event"},
	{Trigger: TriggerControllerProposal, Originates: true, Authority: AuthorityDeclaredPolicy,
		Budget: budgetDeclared, Idempotency: IdempotencyKeyed, Evidence: "proposal-record"},
	{Trigger: TriggerStageWakeup, Originates: true, Authority: AuthorityInherited,
		Budget: BudgetInherited, Idempotency: IdempotencyReplaySafe, Evidence: "stage-receipt"},
	{Trigger: TriggerRetry, Originates: false, Authority: AuthorityInherited,
		Budget: budgetAttemptBounded, Idempotency: IdempotencyKeyed, Evidence: "terminal-attempt-receipt"},
	{Trigger: triggerRerun, Originates: false, Authority: authorityOperatorApproval,
		Budget: budgetInteractive, Idempotency: IdempotencyKeyed, Evidence: "terminal-attempt-receipt"},
	{Trigger: TriggerDelegation, Originates: false, Authority: AuthorityInherited,
		Budget: BudgetInherited, Idempotency: IdempotencyKeyed, Evidence: "delegation-record"},
	{Trigger: TriggerRecovery, Originates: false, Authority: AuthorityDeclaredPolicy,
		Budget: budgetAttemptBounded, Idempotency: IdempotencyReplaySafe, Evidence: "loop-obligation"},
	{Trigger: TriggerFollowup, Originates: false, Authority: AuthorityInherited,
		Budget: BudgetInherited, Idempotency: IdempotencyKeyed, Evidence: "attempt-stimulus"},
}

// TriggerContractFor returns one trigger's declared contract.
func TriggerContractFor(trigger CausalTrigger) (triggerContract, bool) {
	for _, contract := range triggerRegistry {
		if contract.Trigger == trigger {
			return contract, true
		}
	}
	return triggerContract{}, false
}
