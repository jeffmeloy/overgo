package workflowruntime

import (
	"context"
	"errors"
	"maps"
	"slices"
	"sync"
)

// ReconcileCoalescer collapses repeated operational signals — reconnect
// flaps, duplicate wake deliveries, unchanged probes — into one pending
// reconciliation per boundary key. A signal is never lost: it either joins
// the reconciliation that consumes it or leaves its key visibly pending for
// the next one, and one key runs at most one reconciliation at a time.
type ReconcileCoalescer struct {
	mu       sync.Mutex
	pending  map[string]uint64
	inflight map[string]struct{}
}

// NewReconcileCoalescer returns an empty coalescer.
func NewReconcileCoalescer() *ReconcileCoalescer {
	return &ReconcileCoalescer{pending: map[string]uint64{}, inflight: map[string]struct{}{}}
}

// Signal marks one boundary dirty and reports whether the signal coalesced
// into an already-pending reconciliation instead of creating new work.
func (c *ReconcileCoalescer) Signal(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	_, running := c.inflight[key]
	epoch, dirty := c.pending[key]
	c.pending[key] = epoch + 1
	return dirty || running
}

// Reconcile consumes every signal accumulated for one key with exactly one
// run of the reconciler. A signal landing during the run keeps the key
// visibly pending so the next reconciliation observes it, and a second
// caller while one run is inflight returns without running — its signals
// stay pending rather than doubling the work.
func (c *ReconcileCoalescer) Reconcile(
	ctx context.Context,
	key string,
	reconcile func(context.Context) error,
) (bool, error) {
	if reconcile == nil {
		return false, errors.New("workflow runtime: reconciler is required")
	}
	c.mu.Lock()
	epoch, dirty := c.pending[key]
	if !dirty {
		c.mu.Unlock()
		return false, nil
	}
	if _, running := c.inflight[key]; running {
		c.mu.Unlock()
		return false, nil
	}
	c.inflight[key] = struct{}{}
	c.mu.Unlock()
	err := reconcile(ctx)
	c.mu.Lock()
	delete(c.inflight, key)
	if err == nil && c.pending[key] == epoch {
		delete(c.pending, key)
	}
	c.mu.Unlock()
	return true, err
}

// Pending lists boundary keys holding unconsumed signals, so coalesced work
// is visible instead of silently deferred.
func (c *ReconcileCoalescer) Pending() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Sorted(maps.Keys(c.pending))
}
