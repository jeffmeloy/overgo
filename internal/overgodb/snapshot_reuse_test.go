package overgodb

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/processlock"
)

func TestSnapshotReuseAcceptance(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	payload := bytes.Repeat([]byte("snapshot-reuse"), 8192)
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
	if err != nil {
		t.Fatal(err)
	}
	first, err := store.Commit(t.Context(), artifact.Batch{Key: "snapshot/first", Contents: []artifact.Content{{
		Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "application/octet-stream"}, Data: payload,
	}}})
	if err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(t.TempDir(), "backup")
	cold, err := store.Backup(t.Context(), destination)
	if err != nil {
		t.Fatal(err)
	}
	if cold.Head != first || cold.FilesCopied == 0 || cold.BytesCopied < int64(len(payload)) || cold.FilesReused != 0 {
		t.Fatalf("cold: %+v", cold)
	}
	retire := func() {
		t.Helper()
		if err := os.Rename(destination, destination+".partial"); err != nil {
			t.Fatal(err)
		}
	}
	retire()
	warm, err := store.Backup(t.Context(), destination)
	if err != nil {
		t.Fatal(err)
	}
	if warm.Head != first || warm.Extent != cold.Extent || warm.BytesCopied != 0 || warm.FilesCopied != 0 || warm.BytesReused != cold.BytesCopied || warm.FilesReused != cold.FilesCopied {
		t.Fatalf("unchanged snapshot recopied: cold=%+v warm=%+v", cold, warm)
	}
	t.Logf("cold=%+v warm=%+v; store-file counters exclude seal IO and replay", cold, warm)
	if warm.FilesSynced != 0 {
		t.Fatalf("sealed retry resynced files: %+v", warm)
	}
	retire()
	if err := os.Remove(filepath.Join(destination+".partial", "backup-seal.json")); err != nil {
		t.Fatal(err)
	}
	orphan := filepath.Join(destination+".partial", blobDirectory, ".backup-interrupted")
	if err := os.WriteFile(orphan, payload, storeFileMode); err != nil {
		t.Fatal(err)
	}
	unsealed, err := store.Backup(t.Context(), destination)
	if err != nil {
		t.Fatal(err)
	}
	if unsealed.FilesCopied != 0 || unsealed.FilesSynced != cold.FilesCopied {
		t.Fatalf("unsealed bytes were trusted without sync: %+v", unsealed)
	}
	if _, err := os.Stat(filepath.Join(destination, blobDirectory, filepath.Base(orphan))); !os.IsNotExist(err) {
		t.Fatalf("unverified scratch received a seal: %v", err)
	}
	retire()
	second, err := store.Commit(t.Context(), artifact.Batch{Key: "snapshot/second", Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "new-extent")}})
	if err != nil {
		t.Fatal(err)
	}
	advanced, err := store.Backup(t.Context(), destination)
	if err != nil {
		t.Fatal(err)
	}
	if advanced.Head != second || advanced.Sequence != cold.Sequence+1 || advanced.Extent <= cold.Extent || advanced.BytesReused < int64(len(payload)) {
		t.Fatalf("advanced: %+v", advanced)
	}
	retire()
	// Same-size corruption must be replaced, never accepted from metadata.
	candidate, err := newBlobStore(destination + ".partial").path(id)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(candidate, bytes.Repeat([]byte("x"), len(payload)), storeFileMode); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(destination+".partial", storeFilename), []byte("torn"), storeFileMode); err != nil {
		t.Fatal(err)
	}
	repaired, err := store.Backup(t.Context(), destination)
	if err != nil {
		t.Fatal(err)
	}
	if repaired.Head != second || repaired.BytesCopied != advanced.Extent+int64(len(payload)) {
		t.Fatalf("repair: %+v", repaired)
	}
	if err := newBlobStore(destination).verify(id); err != nil {
		t.Fatal(err)
	}
	// Copies remain independent of source file mutations.
	sourceBlob, err := store.blobs.path(id)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sourceBlob, bytes.Repeat([]byte("y"), len(payload)), storeFileMode); err != nil {
		t.Fatal(err)
	}
	if err := newBlobStore(destination).verify(id); err != nil {
		t.Fatalf("snapshot shares source storage: %v", err)
	}
	badDestination := filepath.Join(t.TempDir(), "bad")
	if _, err := store.Backup(t.Context(), badDestination); err == nil {
		t.Fatal("corrupt source published")
	}
	if _, err := os.Stat(badDestination); !os.IsNotExist(err) {
		t.Fatalf("failed snapshot visible: %v", err)
	}
	if err := os.WriteFile(sourceBlob, payload, storeFileMode); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Backup(t.Context(), badDestination); err != nil {
		t.Fatalf("failed publication did not resume: %v", err)
	}

	t.Run("rotation", func(t *testing.T) {
		retire()
		if _, err := store.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		rotated, err := store.Backup(t.Context(), destination)
		if err != nil {
			t.Fatal(err)
		}
		if rotated.Head != second || rotated.BytesReused < int64(len(payload)) {
			t.Fatalf("rotation: %+v", rotated)
		}
	})
	t.Run("contention and cancellation", func(t *testing.T) {
		lock, err := processlock.Acquire(filepath.Join(store.root, lockFilename), storeFileMode)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(nil)
		result := make(chan error, 1)
		go func() { _, err := store.Backup(ctx, filepath.Join(t.TempDir(), "cancelled")); result <- err }()
		cancel(context.Canceled)
		if err := <-result; !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled wait: %v", err)
		}
		started := make(chan struct{})
		go func() {
			close(started)
			_, err := store.Backup(t.Context(), filepath.Join(t.TempDir(), "released"))
			result <- err
		}()
		<-started
		if err := lock.Close(); err != nil {
			t.Fatal(err)
		}
		if err := <-result; err != nil {
			t.Fatalf("released contention failed: %v", err)
		}
	})
	t.Run("cancelled copy resumes", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		path := filepath.Join(t.TempDir(), "cancel")
		var report BackupReport
		if err := report.copyFile(ctx, sourceBlob, path, int64(len(payload)), id.DigestHex(), false); !errors.Is(err, context.Canceled) {
			t.Fatalf("copy cancellation: %v", err)
		}
		if err := report.copyFile(t.Context(), sourceBlob, path, int64(len(payload)), id.DigestHex(), false); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("source aliases are not reusable copies", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "alias")
		if err := os.Link(sourceBlob, path); err != nil {
			t.Fatal(err)
		}
		var report BackupReport
		if err := report.copyFile(t.Context(), sourceBlob, path, int64(len(payload)), id.DigestHex(), true); err != nil {
			t.Fatal(err)
		}
		if report.FilesCopied != 1 || report.FilesReused != 0 {
			t.Fatalf("source hardlink reused: %+v", report)
		}
		copyInfo, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		sourceInfo, err := os.Stat(sourceBlob)
		if err != nil {
			t.Fatal(err)
		}
		if os.SameFile(copyInfo, sourceInfo) {
			t.Fatal("copy still aliases source")
		}
	})
	t.Run("source cannot be staging", func(t *testing.T) {
		root := filepath.Join(t.TempDir(), "backup.partial")
		source, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer source.Close()
		if _, err := source.Commit(t.Context(), fixtureBatch(t)); err != nil {
			t.Fatal(err)
		}
		if _, err := source.Backup(t.Context(), filepath.Join(filepath.Dir(root), "backup")); err == nil {
			t.Fatal("source accepted as mutable staging")
		}
	})
	t.Run("segmented corpus cost", func(t *testing.T) {
		root := t.TempDir()
		buildScaleCorpus(t, root)
		source, err := OpenReadOnly(root)
		if err != nil {
			t.Fatal(err)
		}
		defer source.Close()
		target := filepath.Join(t.TempDir(), "scale")
		cold, err := source.Backup(t.Context(), target)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(target, target+".partial"); err != nil {
			t.Fatal(err)
		}
		warm, err := source.Backup(t.Context(), target)
		if err != nil {
			t.Fatal(err)
		}
		if warm.FilesCopied != 0 || warm.BytesCopied != 0 || warm.BytesReused != cold.BytesCopied {
			t.Fatalf("scale reuse: cold=%+v warm=%+v", cold, warm)
		}
		t.Logf("%d-commit corpus cold=%+v warm=%+v; wall is observational, not a promotion threshold", scaleCorpusCommits, cold, warm)
	})
}
