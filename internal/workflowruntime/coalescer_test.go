package workflowruntime

import (
	"context"
	"errors"
	"sync"
	"testing"
)

// TestCoordinationCoalescesWithoutLostWork pins the coalescing primitive:
// repeated signals for one boundary produce one reconciliation instead of
// one run each, a signal landing during a run stays visibly pending for the
// next one, a concurrent caller never doubles inflight work, and a failed
// reconciliation keeps its signals pending instead of dropping them.
func TestCoordinationCoalescesWithoutLostWork(t *testing.T) {
	ctx := context.Background()
	coalescer := NewReconcileCoalescer()
	if coalesced := coalescer.Signal("boundary"); coalesced {
		t.Fatal("first signal reported an existing pending reconciliation")
	}
	for range 4 {
		if coalesced := coalescer.Signal("boundary"); !coalesced {
			t.Fatal("repeated signal did not coalesce")
		}
	}
	runs := 0
	ran, err := coalescer.Reconcile(ctx, "boundary", func(context.Context) error { runs++; return nil })
	if err != nil || !ran || runs != 1 {
		t.Fatalf("coalesced reconciliation = (ran=%t, %v) runs=%d", ran, err, runs)
	}
	if pending := coalescer.Pending(); len(pending) != 0 {
		t.Fatalf("consumed signals stayed pending: %v", pending)
	}
	if ran, err := coalescer.Reconcile(ctx, "boundary", func(context.Context) error { runs++; return nil }); err != nil || ran {
		t.Fatalf("quiet boundary reconciled = (ran=%t, %v)", ran, err)
	}

	coalescer.Signal("boundary")
	entered, release := make(chan struct{}), make(chan struct{})
	var group sync.WaitGroup
	group.Add(1)
	go func() {
		defer group.Done()
		_, _ = coalescer.Reconcile(ctx, "boundary", func(context.Context) error {
			close(entered)
			<-release
			return nil
		})
	}()
	<-entered
	if coalesced := coalescer.Signal("boundary"); !coalesced {
		t.Fatal("signal during a run did not coalesce")
	}
	if ran, err := coalescer.Reconcile(ctx, "boundary", func(context.Context) error { return nil }); err != nil || ran {
		t.Fatalf("concurrent reconciliation doubled inflight work: (ran=%t, %v)", ran, err)
	}
	close(release)
	group.Wait()
	if pending := coalescer.Pending(); len(pending) != 1 || pending[0] != "boundary" {
		t.Fatalf("mid-run signal was lost: pending=%v", pending)
	}
	mid := 0
	if ran, err := coalescer.Reconcile(ctx, "boundary", func(context.Context) error { mid++; return nil }); err != nil || !ran || mid != 1 {
		t.Fatalf("pending mid-run signal did not reconcile: (ran=%t, %v)", ran, err)
	}

	coalescer.Signal("failing")
	if ran, err := coalescer.Reconcile(ctx, "failing", func(context.Context) error { return errors.New("refused") }); err == nil || !ran {
		t.Fatalf("failing reconciliation = (ran=%t, %v)", ran, err)
	}
	if pending := coalescer.Pending(); len(pending) != 1 || pending[0] != "failing" {
		t.Fatalf("failed reconciliation dropped its signals: %v", pending)
	}
}
