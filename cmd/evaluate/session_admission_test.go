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
	// Cancellation must reach store admission before model resolution: the
	// caller's cause is kept, beside the contention the admission found.
	session, err := openEvaluationSession(ctx, manifest{Repository: root}, modelRequest{Path: "not-loaded.gguf"})
	if session != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled writer admission: session=%v error=%v", session, err)
	}
}
