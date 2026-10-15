package overgodb

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/processlock"
)

// cancelAtJournal cancels startup once replay has created the journal.
type cancelAtJournal struct {
	context.Context
	journal string
	cancel  context.CancelCauseFunc
	cause   error
}

func (c cancelAtJournal) Err() error {
	if _, err := os.Stat(c.journal); err == nil {
		c.cancel(c.cause)
	}
	return c.Context.Err()
}

func TestOpenContextWriterAdmission(t *testing.T) {
	t.Parallel()
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
			lockPath := writerLockPath(root)
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
			// The foreign lock leaves the startup no early success; an early
			// refusal answers the receive below with its error.
			go func() { store, err := OpenContext(ctx, root); done <- result{store, err} }()
			if cancelWait {
				cancel(context.Canceled)
			} else if err := lock.Close(); err != nil {
				t.Fatal(err)
			}
			{
				got := <-done
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
			}
		})
	}
	t.Run("cancel after replay releases startup resources", func(t *testing.T) {
		root := t.TempDir()
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(context.Canceled)
		cause := errors.New("cancel after journal creation")
		store, err := OpenContext(cancelAtJournal{ctx, filepath.Join(root, storeFilename), cancel, cause}, root)
		if store != nil || !errors.Is(err, cause) {
			if store != nil {
				store.Close()
			}
			t.Fatalf("cancelled replay = %v, %v; want no store and caller cause", store, err)
		}
		store, err = Open(root)
		if err != nil {
			t.Fatalf("cancelled replay retained writer admission: %v", err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
	})
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

func TestStoreCloseReleasesWithoutWriterAdmission(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch := fixtureBatch(t)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	store.tail = 0
	journal := store.log.file
	lock, err := processlock.Acquire(writerLockPath(root), storeFileMode)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	for range 2 {
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := journal.ReadAt(make([]byte, 1), 0); !errors.Is(err, os.ErrClosed) {
			t.Fatalf("close retained journal handle: %v", err)
		}
		if other, err := processlock.Acquire(writerLockPath(root), storeFileMode); !errors.Is(err, processlock.ErrBusy) {
			other.Close()
			t.Fatalf("close changed foreign ownership: %v", err)
		}
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	reader, err := OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if _, found, err := reader.Artifact(t.Context(), batch.Artifacts[0].ID); err != nil || !found {
		t.Fatalf("closing without a checkpoint lost committed data: %v, %v", found, err)
	}
}
