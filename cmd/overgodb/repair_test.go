package main

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestRepairRefusesACorruptJournal holds repair to the canonical reader. A
// torn tail, fewer bytes than one frame header, replays read-only without a
// write and is truncated only under -apply. A complete frame whose checksum no
// longer holds, which a scanner that checked only magic, length and JSON called
// clean, a frame resealed over an identity its bytes do not derive, and a
// frame whose length exceeds the reader's limit are refused in both modes, and
// the journal is left byte for byte as it was, with nothing renamed beside it.
func TestRepairRefusesACorruptJournal(t *testing.T) {
	t.Parallel()
	// The journal layout: a store header, then frames of a header that ends
	// in the payload size, the payload and a Castagnoli checksum of both.
	const (
		storeHeader     = 16
		frameHeader     = 84
		frameIdentity   = 48
		framePayloadLen = 80
	)
	root := filepath.Join(t.TempDir(), "store")
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"first commit", "second commit"} {
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "repair/" + text, Contents: []artifact.Content{textContent(t, text)}}); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(root, "overgodb.log")
	clean, err := os.ReadFile(journal)
	if err != nil {
		t.Fatal(err)
	}
	write := func(data []byte) {
		t.Helper()
		if err := os.WriteFile(journal, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	journalIs := func(want []byte, state string) {
		t.Helper()
		if got, err := os.ReadFile(journal); err != nil || !bytes.Equal(got, want) {
			t.Fatalf("after %s the journal holds %d bytes, want %d: %v", state, len(got), len(want), err)
		}
	}
	repair := func(apply bool) error {
		arguments := []string{"repair", "-repo", root}
		if apply {
			arguments = append(arguments, "-apply")
		}
		return runSubcommand(arguments)
	}

	torn := append(slices.Clone(clean), "torn"...)
	write(torn)
	if err := repair(false); err != nil {
		t.Fatalf("a torn tail did not replay: %v", err)
	}
	journalIs(torn, "a read-only repair")
	if err := repair(true); err != nil {
		t.Fatalf("a torn tail was not recovered: %v", err)
	}
	journalIs(clean, "the torn tail's recovery")

	checksum := slices.Clone(clean)
	checksum[len(checksum)-1] ^= 0xFF
	identity := slices.Clone(clean)
	frameEnd := storeHeader + frameHeader + int(binary.LittleEndian.Uint32(identity[storeHeader+framePayloadLen:]))
	identity[storeHeader+frameIdentity] ^= 0xFF
	binary.LittleEndian.PutUint32(identity[frameEnd:], crc32.Checksum(identity[storeHeader:frameEnd], crc32.MakeTable(crc32.Castagnoli)))
	oversized := slices.Clone(clean)
	binary.LittleEndian.PutUint32(oversized[storeHeader+framePayloadLen:], math.MaxUint32)
	for name, damaged := range map[string][]byte{"checksum": checksum, "identity": identity, "oversized": oversized} {
		write(damaged)
		for _, apply := range []bool{false, true} {
			if err := repair(apply); err == nil || !strings.Contains(err.Error(), "restore it from a sealed backup") {
				t.Fatalf("%s damage with apply=%t: repair = %v, want a refusal", name, apply, err)
			}
			journalIs(damaged, name+" damage")
		}
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "overgodb.log.") {
			t.Fatalf("repair left %s beside the journal", entry.Name())
		}
	}
}
