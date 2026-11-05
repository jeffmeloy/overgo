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
	"overgo/internal/testcheck"
)

func TestPublicationStoreLifetime(t *testing.T) {
	repository := t.TempDir()
	record := contractResult(t, "completed measurement")
	otherRecord := contractResult(t, "concurrent writer")
	var retained publishModel
	err := withPublisher(options{Publish: true, Repository: repository}, func(publish publishModel) error {
		retained = publish
		// An admitted idle publisher must leave the database available.
		lock, err := processlock.Acquire(repository+".lock", 0o600)
		if err != nil {
			return err
		}
		defer lock.Close()
		// A publication whose caller has already left carries the cause before
		// attempting a transaction; writer-wait behavior belongs to the store
		// transaction fixture. The acquired lock above proves idle availability.
		left := errors.New("the publication's caller left")
		ctx, leave := context.WithCancelCause(t.Context())
		leave(left)
		if _, err := publish(ctx, record.Inputs.Model, record); !errors.Is(err, left) {
			t.Fatalf("cancelled publication lost its caller cause: %v", err)
		}
		if err := lock.Close(); err != nil {
			return err
		}
		other, err := overgodb.Create(repository)
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
	testcheck.NoError(t, err)
	if _, err := retained(t.Context(), record.Inputs.Model, record); !errors.Is(err, overgodb.ErrClosed) {
		t.Fatalf("publisher outlived command scope: %v", err)
	}
	store, err := overgodb.OpenReadOnly(repository)
	testcheck.NoError(t, err)
	defer store.Close()
	latest := longform.LatestByLocation(t.Context(), store, 0)
	for _, result := range []longform.Result{record, otherRecord} {
		if _, found := latest[longform.Key(result.ModelPath)]; !found {
			t.Fatalf("completed result disappeared: %s", result.ModelPath)
		}
	}
	t.Run("admission precedes measurement", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "not-a-store")
		testcheck.NoError(t, os.WriteFile(path, nil, 0o600))
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
