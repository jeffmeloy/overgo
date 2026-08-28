package workflowruntime

import (
	"testing"

	"overgo/internal/runrecord"
)

// TestExecutionTriggerRegistryCoverage pins the runtime's enqueue
// boundary: every trigger this runtime declares it can fire holds a
// complete registry contract, and the behaviors the runtime's own
// scheduling relies on -- replay-safe wakeups under inherited
// authority, keyed delegation under the delegating execution's budget
// -- are exactly the declared ones.
func TestExecutionTriggerRegistryCoverage(t *testing.T) {
	if err := runrecord.ValidateTriggerRegistry(); err != nil {
		t.Fatal(err)
	}
	if len(ExecutionTriggers) == 0 {
		t.Fatal("runtime declares no execution triggers")
	}
	seen := map[runrecord.CausalTrigger]bool{}
	for _, trigger := range ExecutionTriggers {
		if seen[trigger] {
			t.Fatalf("trigger %q declared twice", trigger)
		}
		seen[trigger] = true
		if _, found := runrecord.TriggerContractFor(trigger); !found {
			t.Fatalf("runtime enqueue trigger %q has no registry contract", trigger)
		}
	}
	wakeup, _ := runrecord.TriggerContractFor(runrecord.TriggerStageWakeup)
	if wakeup.Idempotency != runrecord.IdempotencyReplaySafe || wakeup.Authority != runrecord.AuthorityInherited {
		t.Fatalf("stage wakeup contract = %+v", wakeup)
	}
	delegation, _ := runrecord.TriggerContractFor(runrecord.TriggerDelegation)
	if delegation.Idempotency != runrecord.IdempotencyKeyed || delegation.Budget != runrecord.BudgetInherited {
		t.Fatalf("delegation contract = %+v", delegation)
	}
	webhook, _ := runrecord.TriggerContractFor(runrecord.TriggerWebhook)
	if !seen[runrecord.TriggerWebhook] || webhook.Authority != runrecord.AuthorityDeclaredPolicy ||
		webhook.Idempotency != runrecord.IdempotencyKeyed {
		t.Fatalf("webhook contract = %+v, declared = %v", webhook, seen[runrecord.TriggerWebhook])
	}
}
