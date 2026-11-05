package fsatomic

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestReplacePublishesExactFile(t *testing.T) {
	directory := t.TempDir()
	source := filepath.Join(directory, "source")
	target := filepath.Join(directory, "target")
	testcheck.NoError(t, os.WriteFile(source, []byte("after"), 0o600))
	testcheck.NoError(t, os.WriteFile(target, []byte("before"), 0o600))
	testcheck.NoError(t, Replace(source, target))
	if _, err := os.Stat(source); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("source state = %v, want absent", err)
	}
	got, err := os.ReadFile(target)
	testcheck.NoError(t, err)
	if string(got) != "after" {
		t.Fatalf("target = %q, want after", got)
	}
}

func TestSyncFilePersistsHardLinkMetadata(t *testing.T) {
	directory := t.TempDir()
	source := filepath.Join(directory, "source")
	linked := filepath.Join(directory, "linked")
	testcheck.NoError(t, os.WriteFile(source, []byte("content"), 0o600))
	testcheck.NoError(t, os.Link(source, linked))
	testcheck.NoError(t, SyncFile(linked))
	sourceInfo, err := os.Stat(source)
	testcheck.NoError(t, err)
	linkedInfo, err := os.Stat(linked)
	testcheck.NoError(t, err)
	if !os.SameFile(sourceInfo, linkedInfo) {
		t.Fatal("synced link does not identify its source file")
	}
}

func TestRemoveIsIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "document")
	testcheck.NoError(t, os.WriteFile(path, []byte("content"), 0o600))
	for range 2 {
		testcheck.NoError(t, Remove(path))
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("removed path state = %v, want absent", err)
		}
	}
}
