package objectstore

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

var errStreamFixture = errors.New("stream fixture failed")

type observedRepository struct {
	artifact.Repository
	beforeCommit func(artifact.Batch) error
}

func (repository observedRepository) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	if repository.beforeCommit != nil {
		if err := repository.beforeCommit(batch); err != nil {
			return artifact.CommitID{}, err
		}
	}
	return repository.Repository.Commit(ctx, batch)
}

type zeroReader struct{}

func (zeroReader) Read(buffer []byte) (int, error) { return len(buffer), nil }

type failingReader struct{ remaining int }

func (reader *failingReader) Read(buffer []byte) (int, error) {
	if reader.remaining == 0 {
		return 0, errStreamFixture
	}
	if len(buffer) > reader.remaining {
		buffer = buffer[:reader.remaining]
	}
	reader.remaining -= len(buffer)
	return len(buffer), nil
}

func TestStreamPublicationContract(t *testing.T) {
	t.Run("large stream is installed before catalog publication", func(t *testing.T) {
		ctx := context.Background()
		root := t.TempDir()
		repository, err := overgodb.Open(filepath.Join(root, "repodb"))
		if err != nil {
			t.Fatal(err)
		}
		defer repository.Close()
		objectRoot := filepath.Join(root, "objects")
		observed := observedRepository{Repository: repository, beforeCommit: func(batch artifact.Batch) error {
			for _, location := range batch.Locations {
				info, err := os.Stat(location.Value)
				if err != nil {
					return err
				}
				if !info.Mode().IsRegular() {
					return errors.New("published object is not a regular file")
				}
			}
			return nil
		}}
		store, err := New(objectRoot, observed)
		if err != nil {
			t.Fatal(err)
		}
		streamSize := int64(artifact.MaxContentBytes) + 1
		publication, err := store.Publish(ctx, PublishRequest{
			Kind: artifact.KindModel, MediaType: "application/octet-stream",
			Reader: io.LimitReader(zeroReader{}, streamSize),
		})
		if err != nil {
			t.Fatal(err)
		}
		if publication.Descriptor.Size != uint64(streamSize) || !publication.Commit.Valid() {
			t.Fatalf("publication = %+v", publication)
		}
		descriptor, reader, err := store.Open(ctx, publication.Descriptor.ID)
		if err != nil {
			t.Fatal(err)
		}
		read, copyErr := io.Copy(io.Discard, reader)
		closeErr := reader.Close()
		if copyErr != nil || closeErr != nil || read != streamSize || descriptor != publication.Descriptor {
			t.Fatalf("stream read = (%d, %v, %v), descriptor=%+v", read, copyErr, closeErr, descriptor)
		}
		assertStagingEmpty(t, objectRoot)
	})

	t.Run("retry is content idempotent", func(t *testing.T) {
		ctx := context.Background()
		root := t.TempDir()
		repository, err := overgodb.Open(filepath.Join(root, "repodb"))
		if err != nil {
			t.Fatal(err)
		}
		defer repository.Close()
		store, err := New(filepath.Join(root, "objects"), repository)
		if err != nil {
			t.Fatal(err)
		}
		payload := []byte("idempotent object")
		request := PublishRequest{Kind: artifact.KindDataset, Reader: bytes.NewReader(payload)}
		first, err := store.Publish(ctx, request)
		if err != nil {
			t.Fatal(err)
		}
		request.Reader = bytes.NewReader(payload)
		second, err := store.Publish(ctx, request)
		if err != nil || first != second {
			t.Fatalf("idempotent publications differ: first=%+v second=%+v err=%v", first, second, err)
		}
	})

	t.Run("failed and mismatched streams stay unpublished", func(t *testing.T) {
		ctx := context.Background()
		root := t.TempDir()
		repository, err := overgodb.Open(filepath.Join(root, "repodb"))
		if err != nil {
			t.Fatal(err)
		}
		defer repository.Close()
		objectRoot := filepath.Join(root, "objects")
		store, err := New(objectRoot, repository)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Publish(ctx, PublishRequest{
			Kind: artifact.KindCheckpoint, Reader: &failingReader{remaining: len("partial")},
		}); !errors.Is(err, errStreamFixture) {
			t.Fatalf("failed stream error = %v", err)
		}
		expected, err := artifact.IdentifyBytes(artifact.KindCheckpoint, []byte("expected"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Publish(ctx, PublishRequest{
			Kind: artifact.KindCheckpoint, Expected: &expected, Reader: bytes.NewBufferString("different"),
		}); err == nil {
			t.Fatal("mismatched stream was accepted")
		}
		if _, found, err := repository.Artifact(ctx, expected); err != nil || found {
			t.Fatalf("mismatched identity published = (%v, %v)", found, err)
		}
		assertStagingEmpty(t, objectRoot)
	})
}

func assertStagingEmpty(t *testing.T, root string) {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(root, stagingDirectory))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("staging entries remain: %v", entries)
	}
}
