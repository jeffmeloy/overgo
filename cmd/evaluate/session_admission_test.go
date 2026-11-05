package main

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

func TestEvaluationSessionWriterCancellation(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	testcheck.NoError(t, store.Close())
	lock, err := processlock.Acquire(root+".lock", 0o600)
	testcheck.NoError(t, err)
	defer lock.Close()
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	// Cancellation must reach store admission before model resolution: the
	// caller's cause is kept, beside the contention the admission found.
	session, err := openEvaluationSession(ctx, manifest{Repository: root}, modelRequest{Path: "not-loaded.gguf"})
	if session != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, processlock.ErrBusy) {
		t.Fatalf("cancelled writer admission: session=%v error=%v", session, err)
	}
}
