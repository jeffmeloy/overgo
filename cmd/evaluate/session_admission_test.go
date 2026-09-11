package main

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/processlock"
)

func TestEvaluationSessionWriterCancellation(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	lock, err := processlock.Acquire(filepath.Join(root, "overgodb.lock"), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	// Cancellation must reach store admission before model resolution. The
	// old nonblocking open returned ErrBusy and discarded the caller's cause.
	session, err := openEvaluationSession(ctx, manifest{Repository: root}, modelRequest{Path: "not-loaded.gguf"})
	if session != nil || !errors.Is(err, context.Canceled) || errors.Is(err, processlock.ErrBusy) {
		t.Fatalf("cancelled writer admission: session=%v error=%v", session, err)
	}
}
