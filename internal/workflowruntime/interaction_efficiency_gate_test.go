package workflowruntime

import (
	"context"
	"testing"
)

// TestReconcileCoalescesWakeups proves the coalescing claim from the
// coalescer's own behavior: a flurry of duplicate signals reconciles once
// where the uncoalesced path runs once per signal.
func TestReconcileCoalescesWakeups(t *testing.T) {
	ctx := t.Context()
	coalescer := NewReconcileCoalescer()
	signals := 6
	runs := 0
	for range signals {
		coalescer.Signal("gate-boundary")
	}
	for range signals {
		if _, err := coalescer.Reconcile(ctx, "gate-boundary", func(context.Context) error {
			runs++
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if runs != 1 {
		t.Fatalf("coalesced flurry ran %d reconciliations", runs)
	}
}
