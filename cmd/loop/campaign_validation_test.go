package main

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/authoritylock"
)

func TestCampaignWaitHonorsCancellation(t *testing.T) {
	root := t.TempDir()
	owner, err := authoritylock.Acquire(root)
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if err := authoritylock.Wait(ctx, root); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled wait: %v", err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := authoritylock.Wait(t.Context(), root); err != nil {
		t.Fatal(err)
	}
}
