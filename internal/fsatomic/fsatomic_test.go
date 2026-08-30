package fsatomic

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestReplacePublishesExactFile(t *testing.T) {
	directory := t.TempDir()
	source := filepath.Join(directory, "source")
	target := filepath.Join(directory, "target")
	if err := os.WriteFile(source, []byte("after"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := Replace(source, target); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(source); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("source state = %v, want absent", err)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "after" {
		t.Fatalf("target = %q, want after", got)
	}
}

func TestSyncFilePersistsHardLinkMetadata(t *testing.T) {
	directory := t.TempDir()
	source := filepath.Join(directory, "source")
	linked := filepath.Join(directory, "linked")
	if err := os.WriteFile(source, []byte("content"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(source, linked); err != nil {
		t.Fatal(err)
	}
	if err := SyncFile(linked); err != nil {
		t.Fatal(err)
	}
	sourceInfo, err := os.Stat(source)
	if err != nil {
		t.Fatal(err)
	}
	linkedInfo, err := os.Stat(linked)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(sourceInfo, linkedInfo) {
		t.Fatal("synced link does not identify its source file")
	}
}

func TestRemoveIsIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "document")
	if err := os.WriteFile(path, []byte("content"), 0o600); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := Remove(path); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("removed path state = %v, want absent", err)
		}
	}
}
