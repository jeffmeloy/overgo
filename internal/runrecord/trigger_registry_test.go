package runrecord

import (
	"errors"
	"fmt"
	"slices"
	"testing"
)

// TestExecutionTriggerRegistryCoverage pins the registry contract:
// every canonical causal trigger declares a complete contract, the
// declared behaviors match the substrate they bind to, and mutations
// that would open an undocumented enqueue path fail validation.
func TestExecutionTriggerRegistryCoverage(t *testing.T) {
	if err := validateTriggerRegistry(); err != nil {
		t.Fatal(err)
	}
	canonical := append(slices.Clone(originatingTriggers), derivedTriggers...)
	for _, trigger := range canonical {
		contract, found := TriggerContractFor(trigger)
		if !found {
			t.Fatalf("trigger %q has no contract", trigger)
		}
		if contract.Originates != slices.Contains(originatingTriggers, trigger) {
			t.Fatalf("trigger %q origination differs from its class", trigger)
		}
	}
	if _, found := TriggerContractFor("cosmic-ray"); found {
		t.Fatal("foreign trigger has a contract")
	}

	// The declared bindings this substrate depends on.
	retry, _ := TriggerContractFor(TriggerRetry)
	if retry.Budget != budgetAttemptBounded || retry.Evidence != "terminal-attempt-receipt" {
		t.Fatalf("retry contract = %+v", retry)
	}
	recovery, _ := TriggerContractFor(TriggerRecovery)
	if recovery.Idempotency != IdempotencyReplaySafe || recovery.Evidence != "loop-obligation" {
		t.Fatalf("recovery contract = %+v", recovery)
	}
	manual, _ := TriggerContractFor(TriggerManual)
	if manual.Authority != authorityOperatorApproval || manual.Idempotency != idempotencyNone {
		t.Fatalf("manual contract = %+v", manual)
	}

	// Every mutation that would open an undocumented path refuses.
	pristine := triggerRegistry
	defer func() { triggerRegistry = pristine }()
	mutations := []struct {
		name   string
		mutate func()
	}{
		{"missing trigger", func() { triggerRegistry = pristine[1:] }},
		{"duplicated trigger", func() {
			triggerRegistry = append(slices.Clone(pristine[:len(pristine)-1]), pristine[0])
		}},
		{"foreign trigger", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Trigger = "cosmic-ray"
			triggerRegistry = mutated
		}},
		{"origination against its class", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Originates = !mutated[0].Originates
			triggerRegistry = mutated
		}},
		{"authority outside the vocabulary", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Authority = "vibes"
			triggerRegistry = mutated
		}},
		{"budget outside the vocabulary", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Budget = "unlimited"
			triggerRegistry = mutated
		}},
		{"idempotency outside the vocabulary", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Idempotency = "hopefully"
			triggerRegistry = mutated
		}},
		{"empty evidence", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Evidence = ""
			triggerRegistry = mutated
		}},
		{"idempotency skipped outside interactive admission", func() {
			mutated := slices.Clone(pristine)
			for index := range mutated {
				if mutated[index].Trigger == triggerSchedule {
					mutated[index].Idempotency = idempotencyNone
				}
			}
			triggerRegistry = mutated
		}},
	}
	for _, mutation := range mutations {
		mutation.mutate()
		if err := validateTriggerRegistry(); err == nil {
			t.Fatalf("%s: accepted", mutation.name)
		}
		triggerRegistry = pristine
	}
}

// validateTriggerRegistry refuses an incomplete or inconsistent
// registry: every canonical trigger appears exactly once, every field
// is drawn from its closed vocabulary, origination agrees with the
// causal trigger classes, and only interactive admission may skip
// idempotency.
func validateTriggerRegistry() error {
	canonical := append(slices.Clone(originatingTriggers), derivedTriggers...)
	if len(triggerRegistry) != len(canonical) {
		return errors.New("run record: trigger registry does not cover the causal vocabulary")
	}
	seen := map[CausalTrigger]bool{}
	for _, contract := range triggerRegistry {
		if seen[contract.Trigger] || !slices.Contains(canonical, contract.Trigger) {
			return fmt.Errorf("run record: trigger %q is foreign or declared twice", contract.Trigger)
		}
		seen[contract.Trigger] = true
		if contract.Originates != slices.Contains(originatingTriggers, contract.Trigger) {
			return fmt.Errorf("run record: trigger %q origination disagrees with its causal class", contract.Trigger)
		}
		if !slices.Contains([]triggerAuthority{authorityOperatorApproval, AuthorityDeclaredPolicy, AuthorityInherited}, contract.Authority) ||
			!slices.Contains([]triggerBudget{budgetInteractive, budgetDeclared, budgetAttemptBounded, BudgetInherited}, contract.Budget) ||
			!slices.Contains([]triggerIdempotency{idempotencyNone, IdempotencyKeyed, IdempotencyReplaySafe}, contract.Idempotency) ||
			!validLabel(contract.Evidence) {
			return fmt.Errorf("run record: trigger %q contract leaves its vocabulary", contract.Trigger)
		}
		if contract.Idempotency == idempotencyNone && contract.Budget != budgetInteractive {
			return fmt.Errorf("run record: trigger %q skips idempotency outside interactive admission", contract.Trigger)
		}
	}
	return nil
}
