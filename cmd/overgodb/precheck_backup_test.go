package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestPrecheckTakesItsOwnBackup holds the command to sealing the backup it
// works from: the backup is named for the head and sequence it seals and its
// seal says the same, a second call at that head finds it rather than copying
// the store again, and a call after the head has moved seals another.
func TestPrecheckTakesItsOwnBackup(t *testing.T) {
	t.Parallel()
	repository, backups := filepath.Join(t.TempDir(), "store"), t.TempDir()
	commit := func(text string) {
		store, err := overgodb.Open(repository)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		data := []byte(strings.Repeat(text, 1<<13))
		id, err := artifact.IdentifyBytes(artifact.KindEvidence, data)
		if err != nil {
			t.Fatal(err)
		}
		content := artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: "text/plain"}, Data: data}
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "backup/" + text, Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
	}
	commit("first")
	first, err := sealBackup(t.Context(), repository, backups)
	if err != nil {
		t.Fatal(err)
	}
	seal, err := overgodb.ReadBackupSeal(first)
	if err != nil || !strings.Contains(filepath.Base(first), seal.Head.String()[:12]) {
		t.Fatalf("backup %s does not carry the head its seal names: %+v, %v", first, seal, err)
	}
	if again, err := sealBackup(t.Context(), repository, backups); err != nil || again != first {
		t.Fatalf("a second seal at the same head = %s, %v; want the backup already there", again, err)
	}
	commit("second")
	if moved, err := sealBackup(t.Context(), repository, backups); err != nil || moved == first {
		t.Fatalf("a seal after the head moved = %s, %v; want a backup of its own", moved, err)
	}
}
