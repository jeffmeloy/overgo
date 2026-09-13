package overgodb

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/processlock"
)

// Pause at the first cancellation check after the blob destination exists.
// The caller can inspect ownership without a timer or production test hook.
type backupBlobBoundary struct {
	context.Context
	directory       string
	sourceLock      string
	cancel          context.CancelCauseFunc
	metadataChecked bool
	metadataErr     error
	entered         chan struct{}
	resume          chan struct{}
	pause           func()
}

func (c *backupBlobBoundary) Err() error {
	if _, err := os.Stat(c.directory); err == nil {
		c.pause()
	} else if _, err := os.Stat(filepath.Dir(c.directory)); err == nil {
		c.metadataChecked = true
		lock, err := processlock.Acquire(c.sourceLock, storeFileMode)
		if err == nil {
			c.metadataErr = errors.Join(errors.New("mutable backup metadata was not pinned"), lock.Close())
			c.cancel(c.metadataErr)
		} else if !errors.Is(err, processlock.ErrBusy) {
			c.metadataErr = err
			c.cancel(c.metadataErr)
		}
	}
	return c.Context.Err()
}

func TestBackupWriterReleaseAcceptance(t *testing.T) {
	for _, cancelCopy := range []bool{false, true} {
		name := "commit-and-rotate"
		if cancelCopy {
			name = "cancel-and-resume"
		}
		t.Run(name, func(t *testing.T) {
			store, err := Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			payload := bytes.Repeat([]byte("captured immutable blob"), 8192)
			id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(t.Context(), artifact.Batch{Key: "backup/captured", Contents: []artifact.Content{{
				Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "application/octet-stream"}, Data: payload,
			}}}); err != nil {
				t.Fatal(err)
			}
			if _, err := store.Snapshot(t.Context()); err != nil {
				t.Fatal(err)
			}
			head, sequence := store.Head()
			writer, err := Open(store.root)
			if err != nil {
				t.Fatal(err)
			}
			defer writer.Close()
			destination := filepath.Join(t.TempDir(), "backup")
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			boundary := &backupBlobBoundary{Context: ctx, directory: filepath.Join(destination+".partial", blobDirectory), sourceLock: filepath.Join(store.root, lockFilename), cancel: cancel, entered: make(chan struct{}), resume: make(chan struct{})}
			boundary.pause = sync.OnceFunc(func() {
				close(boundary.entered)
				<-boundary.resume
			})
			type result struct {
				report *BackupReport
				err    error
			}
			done := make(chan result, 1)
			go func() {
				report, err := store.Backup(boundary, destination)
				done <- result{report, err}
			}()
			select {
			case <-boundary.entered:
			case result := <-done:
				t.Fatalf("backup ended before the blob boundary: %v; metadata protection: %v", result.err, boundary.metadataErr)
			}
			// A failed assertion must still release the paused worker.
			resume := sync.OnceFunc(func() { close(boundary.resume) })
			finished := false
			defer func() {
				cancel(nil)
				resume()
				if !finished {
					<-done
				}
			}()
			if !boundary.metadataChecked || boundary.metadataErr != nil {
				t.Fatalf("mutable metadata protection was not observed: %v", boundary.metadataErr)
			}
			lock, err := processlock.Acquire(filepath.Join(store.root, lockFilename), storeFileMode)
			if err != nil {
				cancel(nil)
				resume()
				<-done
				finished = true
				t.Fatalf("writer unavailable during immutable blob copy: %v", err)
			}
			if err := lock.Close(); err != nil {
				t.Fatal(err)
			}
			tail := fixtureDescriptor(t, artifact.KindOutput, "post-capture writer")
			if cancelCopy {
				cancel(nil)
			} else {
				if _, err := writer.Commit(t.Context(), artifact.Batch{Key: "backup/post-capture", Artifacts: []artifact.Descriptor{tail}}); err != nil {
					t.Fatal(err)
				}
				if _, err := writer.Snapshot(t.Context()); err != nil {
					t.Fatal(err)
				}
			}
			resume()
			copied := <-done
			finished = true
			if cancelCopy {
				if !errors.Is(copied.err, context.Canceled) {
					t.Fatalf("copy cancellation: %v", copied.err)
				}
				if _, err := os.Stat(destination); !os.IsNotExist(err) {
					t.Fatalf("cancelled copy was published: %v", err)
				}
				copied.report, copied.err = store.Backup(t.Context(), destination)
			}
			if copied.err != nil || copied.report.Head != head || copied.report.Sequence != sequence {
				t.Fatalf("captured identity changed: %+v, %v", copied.report, copied.err)
			}
			replica, err := OpenReadOnly(destination)
			if err != nil {
				t.Fatal(err)
			}
			defer replica.Close()
			if _, found, err := replica.Artifact(t.Context(), tail.ID); err != nil || found {
				t.Fatalf("post-capture content entered backup: found=%t err=%v", found, err)
			}
			_, reader, found, err := replica.OpenContent(t.Context(), id)
			if err != nil || !found {
				t.Fatalf("captured blob missing: found=%t err=%v", found, err)
			}
			actual, err := io.ReadAll(reader)
			if err != nil || !bytes.Equal(actual, payload) {
				t.Fatalf("captured blob changed: %v", err)
			}
		})
	}
}
