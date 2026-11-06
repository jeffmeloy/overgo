package main

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/authoritylock"
	"overgo/internal/testcheck"
)

func TestCampaignWaitHonorsCancellation(t *testing.T) {
	root := t.TempDir()
	owner := testcheck.Must(authoritylock.Acquire(root))(t)
	defer owner.Close()
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if err := authoritylock.Wait(ctx, root); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled wait: %v", err)
	}
	testcheck.NoError(t, owner.Close())
	testcheck.NoError(t, authoritylock.Wait(t.Context(), root))
}
