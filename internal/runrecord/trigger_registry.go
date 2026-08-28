package runrecord

import (
	"errors"
	"fmt"
	"slices"
)

// TriggerAuthority names where a trigger's permission to execute comes
// from. Authority never derives from the causal chain itself.
type TriggerAuthority string

const (
	// AuthorityOperatorApproval requires an explicit operator decision.
	AuthorityOperatorApproval TriggerAuthority = "operator-approval"
	// AuthorityDeclaredPolicy executes under a committed policy document.
	AuthorityDeclaredPolicy TriggerAuthority = "declared-policy"
	// AuthorityInherited executes under the admitting execution's own
	// authority, which must already exist.
	AuthorityInherited TriggerAuthority = "inherited"
)

// TriggerBudget names which budget admits a trigger's execution.
type TriggerBudget string

const (
	// BudgetInteractive charges the interactive session budget.
	BudgetInteractive TriggerBudget = "interactive"
	// BudgetDeclared charges a budget a committed policy declares.
	BudgetDeclared TriggerBudget = "declared"
	// BudgetAttemptBounded charges the bounded attempt budget.
	BudgetAttemptBounded TriggerBudget = "attempt-bounded"
	// BudgetInherited charges the admitting execution's budget.
	BudgetInherited TriggerBudget = "inherited"
)

// TriggerIdempotency names how duplicate firings are prevented.
type TriggerIdempotency string

const (
	// IdempotencyNone admits every firing; only interactive triggers may
	// declare it.
	IdempotencyNone TriggerIdempotency = "none"
	// IdempotencyKeyed dedupes firings by an explicit key.
	IdempotencyKeyed TriggerIdempotency = "keyed"
	// IdempotencyReplaySafe converges under repetition by construction.
	IdempotencyReplaySafe TriggerIdempotency = "replay-safe"
)

// TriggerContract is one trigger kind's complete declared behavior:
// how it participates in causality, whose authority admits it, which
// budget it charges, how duplicates are prevented, and the evidence
// record that must exist when it fires.
type TriggerContract struct {
	Trigger CausalTrigger
	// Originates reports whether the trigger mints a causal root.
	Originates  bool
	Authority   TriggerAuthority
	Budget      TriggerBudget
	Idempotency TriggerIdempotency
	// Evidence names the record kind that must exist for the firing.
	Evidence string
}

// TriggerRegistry declares every canonical trigger's contract, in
// CausalTrigger declaration order. There is no undocumented enqueue
// path: a trigger added to the causal vocabulary without a complete
// contract here fails ValidateTriggerRegistry and the architecture
// tests built on it.
var TriggerRegistry = []TriggerContract{
	{Trigger: TriggerManual, Originates: true, Authority: AuthorityOperatorApproval,
		Budget: BudgetInteractive, Idempotency: IdempotencyNone, Evidence: "operator-decision"},
	{Trigger: TriggerSchedule, Originates: true, Authority: AuthorityDeclaredPolicy,
		Budget: BudgetDeclared, Idempotency: IdempotencyKeyed, Evidence: "schedule-slot"},
	{Trigger: TriggerWebhook, Originates: true, Authority: AuthorityDeclaredPolicy,
		Budget: BudgetDeclared, Idempotency: IdempotencyKeyed, Evidence: "ingress-event"},
	{Trigger: TriggerControllerProposal, Originates: true, Authority: AuthorityDeclaredPolicy,
		Budget: BudgetDeclared, Idempotency: IdempotencyKeyed, Evidence: "proposal-record"},
	{Trigger: TriggerStageWakeup, Originates: true, Authority: AuthorityInherited,
		Budget: BudgetInherited, Idempotency: IdempotencyReplaySafe, Evidence: "stage-receipt"},
	{Trigger: TriggerRetry, Originates: false, Authority: AuthorityInherited,
		Budget: BudgetAttemptBounded, Idempotency: IdempotencyKeyed, Evidence: "terminal-attempt-receipt"},
	{Trigger: TriggerRerun, Originates: false, Authority: AuthorityOperatorApproval,
		Budget: BudgetInteractive, Idempotency: IdempotencyKeyed, Evidence: "terminal-attempt-receipt"},
	{Trigger: TriggerDelegation, Originates: false, Authority: AuthorityInherited,
		Budget: BudgetInherited, Idempotency: IdempotencyKeyed, Evidence: "delegation-record"},
	{Trigger: TriggerRecovery, Originates: false, Authority: AuthorityDeclaredPolicy,
		Budget: BudgetAttemptBounded, Idempotency: IdempotencyReplaySafe, Evidence: "loop-obligation"},
}

// TriggerContractFor returns one trigger's declared contract.
func TriggerContractFor(trigger CausalTrigger) (TriggerContract, bool) {
	for _, contract := range TriggerRegistry {
		if contract.Trigger == trigger {
			return contract, true
		}
	}
	return TriggerContract{}, false
}

// ValidateTriggerRegistry refuses an incomplete or inconsistent
// registry: every canonical trigger appears exactly once, every field
// is drawn from its closed vocabulary, origination agrees with the
// causal trigger classes, and only interactive admission may skip
// idempotency.
func ValidateTriggerRegistry() error {
	canonical := append(slices.Clone(OriginatingTriggers), DerivedTriggers...)
	if len(TriggerRegistry) != len(canonical) {
		return errors.New("run record: trigger registry does not cover the causal vocabulary")
	}
	seen := map[CausalTrigger]bool{}
	for _, contract := range TriggerRegistry {
		if seen[contract.Trigger] || !slices.Contains(canonical, contract.Trigger) {
			return fmt.Errorf("run record: trigger %q is foreign or declared twice", contract.Trigger)
		}
		seen[contract.Trigger] = true
		if contract.Originates != slices.Contains(OriginatingTriggers, contract.Trigger) {
			return fmt.Errorf("run record: trigger %q origination disagrees with its causal class", contract.Trigger)
		}
		if !slices.Contains([]TriggerAuthority{AuthorityOperatorApproval, AuthorityDeclaredPolicy, AuthorityInherited}, contract.Authority) ||
			!slices.Contains([]TriggerBudget{BudgetInteractive, BudgetDeclared, BudgetAttemptBounded, BudgetInherited}, contract.Budget) ||
			!slices.Contains([]TriggerIdempotency{IdempotencyNone, IdempotencyKeyed, IdempotencyReplaySafe}, contract.Idempotency) ||
			!validLabel(contract.Evidence) {
			return fmt.Errorf("run record: trigger %q contract leaves its vocabulary", contract.Trigger)
		}
		if contract.Idempotency == IdempotencyNone && contract.Budget != BudgetInteractive {
			return fmt.Errorf("run record: trigger %q skips idempotency outside interactive admission", contract.Trigger)
		}
	}
	return nil
}
