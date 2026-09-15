package workflowruntime

import (
	"context"
	"errors"
	"testing"
	"time"

	"overgo/internal/runrecord"
)

func TestStageProgressPrecedesSiblingCompletion(t *testing.T) {
	fixture := newParallelFixture(t)
	// A deadlock watchdog, not a throughput requirement: the right adapter
	// cannot complete until the left receipt is published and observed.
	ctx, cancel := context.WithTimeoutCause(t.Context(), 5*time.Second, errors.New("durable progress was not delivered"))
	defer cancel()
	release := make(chan struct{})
	observed := make(chan error, 1)
	fixture.runtime.ObserveStages(func(receipt runrecord.StageReceipt) {
		if receipt.Node != "left" {
			return
		}
		stored, found, err := runrecord.ResolveStageReceipt(ctx, fixture.store, fixture.operation, "left")
		if err == nil && (!found || stored.ID != receipt.ID || stored.State != runrecord.StageCompleted) {
			err = errors.New("progress preceded durable receipt")
		}
		observed <- err
		close(release)
	})
	registerParallelAdapters(t, fixture, func(ctx context.Context, value string) (string, error) {
		if value == "right" {
			select {
			case <-release:
			case <-ctx.Done():
				return "", context.Cause(ctx)
			}
		}
		return value, nil
	})
	if _, err := fixture.execute(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-observed; err != nil {
		t.Fatal(err)
	}
}
