package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
)

func TestPublicationStoreLifetime(t *testing.T) {
	repository := t.TempDir()
	record := contractResult(t, "completed measurement")
	otherRecord := contractResult(t, "concurrent writer")
	var retained publishModel
	err := withPublisher(options{Publish: true, Repository: repository}, func(publish publishModel) error {
		retained = publish
		// An admitted idle publisher must leave the database available.
		lock, err := processlock.Acquire(filepath.Join(repository, "overgodb.lock"), 0o600)
		if err != nil {
			return err
		}
		defer lock.Close()
		// A publication whose caller has left waits for no writer and carries
		// the caller's cause; the lock beside it proves the contention.
		left := errors.New("the publication's caller left")
		ctx, leave := context.WithCancelCause(t.Context())
		leave(left)
		if _, err := publish(ctx, record.Inputs.Model, record); !errors.Is(err, left) {
			t.Fatalf("publication must wait cancellably, not reopen and fail immediately: %v", err)
		}
		if err := lock.Close(); err != nil {
			return err
		}
		other, err := overgodb.Open(repository)
		if err != nil {
			return err
		}
		_, publishErr := longform.Publish(t.Context(), other, otherRecord.Inputs.Model, otherRecord)
		if err := errors.Join(publishErr, other.Close()); err != nil {
			return err
		}
		// Reuse the same completed result; the transaction refresh retains both writes.
		_, err = publish(t.Context(), record.Inputs.Model, record)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := retained(t.Context(), record.Inputs.Model, record); !errors.Is(err, overgodb.ErrClosed) {
		t.Fatalf("publisher outlived command scope: %v", err)
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, result := range []longform.Result{record, otherRecord} {
		if _, found := longform.Latest(t.Context(), store, result.ModelPath); !found {
			t.Fatalf("completed result disappeared: %s", result.ModelPath)
		}
	}
	t.Run("admission precedes measurement", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "not-a-store")
		if err := os.WriteFile(path, nil, 0o600); err != nil {
			t.Fatal(err)
		}
		called := false
		if err := withPublisher(options{Publish: true, Repository: path}, func(publishModel) error { called = true; return nil }); err == nil || called {
			t.Fatalf("measurement started without publication admission: called=%t err=%v", called, err)
		}
		if err := withPublisher(options{Repository: path}, func(publish publishModel) error {
			if publish != nil {
				t.Fatal("dry run admitted a writer")
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	})
}
