package overgodb

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/processlock"
)

func TestOpenContextWriterAdmission(t *testing.T) {
	for _, cancelWait := range []bool{false, true} {
		t.Run(map[bool]string{false: "release resumes opening", true: "cancel leaves owner intact"}[cancelWait], func(t *testing.T) {
			root := t.TempDir()
			owner, err := Open(root)
			if err != nil {
				t.Fatal(err)
			}
			batch := fixtureBatch(t)
			if _, err := owner.Commit(t.Context(), batch); err != nil {
				t.Fatal(err)
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			lockPath := filepath.Join(root, lockFilename)
			lock, err := processlock.Acquire(lockPath, storeFileMode)
			if err != nil {
				t.Fatal(err)
			}
			defer lock.Close()
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			type result struct {
				store *Store
				err   error
			}
			done := make(chan result, 1)
			go func() { store, err := OpenContext(ctx, root); done <- result{store, err} }()
			select {
			case got := <-done:
				if got.store != nil {
					got.store.Close()
				}
				t.Fatalf("startup returned before release: %v", got.err)
			case <-time.After(20 * time.Millisecond):
			}
			if cancelWait {
				cancel(context.Canceled)
			} else if err := lock.Close(); err != nil {
				t.Fatal(err)
			}
			select {
			case got := <-done:
				if cancelWait {
					if got.store != nil {
						got.store.Close()
						t.Fatal("cancelled startup returned a store")
					}
					if !errors.Is(got.err, context.Canceled) {
						t.Fatalf("cancelled startup: %v", got.err)
					}
					if other, err := processlock.Acquire(lockPath, storeFileMode); !errors.Is(err, processlock.ErrBusy) {
						other.Close()
						t.Fatalf("cancelled startup changed foreign ownership: %v", err)
					}
					return
				}
				if got.err != nil {
					t.Fatal(got.err)
				}
				defer got.store.Close()
				if _, found, err := got.store.Artifact(t.Context(), batch.Artifacts[0].ID); err != nil || !found {
					t.Fatalf("admitted store lost committed data: %v, %v", found, err)
				}
				if next, err := Open(root); err != nil {
					t.Fatalf("idle admitted handle retained lock: %v", err)
				} else {
					next.Close()
				}
			case <-time.After(5 * time.Second):
				t.Fatal("startup did not respond to release or cancellation")
			}
		})
	}
	t.Run("invalid context and path", func(t *testing.T) {
		if _, err := OpenContext(nil, t.TempDir()); err == nil {
			t.Fatal("nil context admitted")
		}
		if _, err := OpenContext(t.Context(), ""); err == nil || errors.Is(err, processlock.ErrBusy) {
			t.Fatalf("invalid root treated as contention: %v", err)
		}
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		if _, err := OpenContext(ctx, t.TempDir()); !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-cancelled startup: %v", err)
		}
	})
}
