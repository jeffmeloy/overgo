package runrecord

import (
	"slices"
	"testing"
)

// TestExecutionTriggerRegistryCoverage pins the registry contract:
// every canonical causal trigger declares a complete contract, the
// declared behaviors match the substrate they bind to, and mutations
// that would open an undocumented enqueue path fail validation.
func TestExecutionTriggerRegistryCoverage(t *testing.T) {
	if err := ValidateTriggerRegistry(); err != nil {
		t.Fatal(err)
	}
	canonical := append(slices.Clone(OriginatingTriggers), DerivedTriggers...)
	for _, trigger := range canonical {
		contract, found := TriggerContractFor(trigger)
		if !found {
			t.Fatalf("trigger %q has no contract", trigger)
		}
		if contract.Originates != slices.Contains(OriginatingTriggers, trigger) {
			t.Fatalf("trigger %q origination differs from its class", trigger)
		}
	}
	if _, found := TriggerContractFor("cosmic-ray"); found {
		t.Fatal("foreign trigger has a contract")
	}

	// The declared bindings this substrate depends on.
	retry, _ := TriggerContractFor(TriggerRetry)
	if retry.Budget != BudgetAttemptBounded || retry.Evidence != "terminal-attempt-receipt" {
		t.Fatalf("retry contract = %+v", retry)
	}
	recovery, _ := TriggerContractFor(TriggerRecovery)
	if recovery.Idempotency != IdempotencyReplaySafe || recovery.Evidence != "loop-obligation" {
		t.Fatalf("recovery contract = %+v", recovery)
	}
	manual, _ := TriggerContractFor(TriggerManual)
	if manual.Authority != AuthorityOperatorApproval || manual.Idempotency != IdempotencyNone {
		t.Fatalf("manual contract = %+v", manual)
	}

	// Every mutation that would open an undocumented path refuses.
	pristine := TriggerRegistry
	defer func() { TriggerRegistry = pristine }()
	mutations := []struct {
		name   string
		mutate func()
	}{
		{"missing trigger", func() { TriggerRegistry = pristine[1:] }},
		{"duplicated trigger", func() {
			TriggerRegistry = append(slices.Clone(pristine[:len(pristine)-1]), pristine[0])
		}},
		{"foreign trigger", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Trigger = "cosmic-ray"
			TriggerRegistry = mutated
		}},
		{"origination against its class", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Originates = !mutated[0].Originates
			TriggerRegistry = mutated
		}},
		{"authority outside the vocabulary", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Authority = "vibes"
			TriggerRegistry = mutated
		}},
		{"budget outside the vocabulary", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Budget = "unlimited"
			TriggerRegistry = mutated
		}},
		{"idempotency outside the vocabulary", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Idempotency = "hopefully"
			TriggerRegistry = mutated
		}},
		{"empty evidence", func() {
			mutated := slices.Clone(pristine)
			mutated[0].Evidence = ""
			TriggerRegistry = mutated
		}},
		{"idempotency skipped outside interactive admission", func() {
			mutated := slices.Clone(pristine)
			for index := range mutated {
				if mutated[index].Trigger == TriggerSchedule {
					mutated[index].Idempotency = IdempotencyNone
				}
			}
			TriggerRegistry = mutated
		}},
	}
	for _, mutation := range mutations {
		mutation.mutate()
		if err := ValidateTriggerRegistry(); err == nil {
			t.Fatalf("%s: accepted", mutation.name)
		}
		TriggerRegistry = pristine
	}
}
