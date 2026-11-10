//go:build windows

package executor

import (
	"context"
	"runtime"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestAReleaseBesideACloseFinishes orders the two the review read in two lock orders (executor-close/release-race): a
// resident session's release already on the worker's turn, which drops the executor's graph execs, and a Close that
// holds the executor's lock while it waits for the worker. The release is let on only once the Close holds the lock.
// Both return, and the executor is closed.
// The worker is closed only once they have returned, since a worker left waiting would hold its close too.
func TestAReleaseBesideACloseFinishes(t *testing.T) {
	cudatest.Require(t)
	// The two are given half the run's remaining time, so two that wait on each other fail here, on an assertion,
	// before the run's own deadline ends it with no test named.
	ctx := t.Context()
	if deadline, bounded := t.Deadline(); bounded {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, time.Now().Add(time.Until(deadline)/2))
		defer cancel()
	}
	worker := testcheck.Must(device.New(cudaFixtureDevice))(t)
	executor := testcheck.Must(NewWithWorker(worker))(t)
	onTurn, gate := make(chan struct{}), make(chan struct{})
	released, closed := make(chan error, 1), make(chan error, 1)
	go func() {
		released <- worker.Do(context.Background(), func(state *device.State) error {
			close(onTurn)
			<-gate
			return executor.DropGraphExecs(state)
		})
	}()
	<-onTurn
	go func() { closed <- executor.Close() }()
	// The Close holds the executor's lock while it waits for the worker the release holds.
	for executor.mu.TryRLock() {
		executor.mu.RUnlock()
		runtime.Gosched()
	}
	close(gate)
	for range 2 {
		select {
		case err := <-released:
			testcheck.NoError(t, err)
		case err := <-closed:
			testcheck.NoError(t, err)
		case <-ctx.Done():
			t.Fatal("a release on the worker's turn and a Close holding the executor's lock waited on each other for half the run's time")
		}
	}
	if !executor.closed {
		t.Error("the executor is not closed")
	}
	testcheck.NoError(t, worker.Close())
}
