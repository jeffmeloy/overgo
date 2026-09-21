package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestRetirementReportNamesWhatIsSafe holds the leftovers report to what it
// is for: it finds the stores a swap superseded and the backups a seal
// proves, ignores a directory under the backups that is no backup, marks the
// newest of each kind to keep and the rest not, says which hold the packed
// layout, and removes nothing.
func TestRetirementReportNamesWhatIsSafe(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	repository, backups := filepath.Join(root, "store"), filepath.Join(root, "backups")
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
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "leftovers/" + text, Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
	}
	commit("first")
	older, err := sealBackup(t.Context(), repository, backups)
	if err != nil {
		t.Fatal(err)
	}
	commit("second")
	newer, err := sealBackup(t.Context(), repository, backups)
	if err != nil {
		t.Fatal(err)
	}
	stale, fresh := repository+".superseded-aaaaaaaaaaaa", repository+".superseded-bbbbbbbbbbbb"
	for _, directory := range []string{stale, filepath.Join(fresh, "blobs"), filepath.Join(backups, "not-a-backup")} {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(fresh, "blobs", "small.pack"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(stale, time.Time{}, time.Now().Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}

	found, err := listLeftovers(repository, backups)
	if err != nil {
		t.Fatal(err)
	}
	want := []leftover{
		{Path: newer, Kind: leftoverBackup, Keep: true},
		{Path: older, Kind: leftoverBackup},
		{Path: fresh, Kind: leftoverSuperseded, Packed: true, Keep: true},
		{Path: stale, Kind: leftoverSuperseded},
	}
	if len(found) != len(want) {
		t.Fatalf("leftovers = %+v, want %d of them", found, len(want))
	}
	for index, entry := range found {
		if entry.Path != want[index].Path || entry.Kind != want[index].Kind || entry.Keep != want[index].Keep || entry.Packed != want[index].Packed {
			t.Fatalf("leftover %d = %+v, want %+v", index, entry, want[index])
		}
		if _, err := os.Stat(entry.Path); err != nil {
			t.Fatalf("the report removed %s: %v", entry.Path, err)
		}
	}
}
