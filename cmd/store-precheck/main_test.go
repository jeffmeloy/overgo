package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
)

// TestFailedCheckNeverSwaps holds the verdict to its rule: every check runs
// and reports, one failure or no check at all refuses, a refusal never
// reaches the swap, and a clean verdict swaps only when asked.
func TestFailedCheckNeverSwaps(t *testing.T) {
	ran := 0
	verdicts := runChecks([]namedCheck{
		{"first", func() error { ran++; return errors.New("no successful gate attempt") }},
		{"second", func() error { ran++; return nil }},
	})
	if ran != 2 || verdicts[0].Passed || verdicts[0].Detail == "" || !verdicts[1].Passed {
		t.Fatalf("checks ran=%d verdicts=%+v", ran, verdicts)
	}
	swaps := 0
	swap := func() (string, error) { swaps++; return "superseded", nil }
	for _, refused := range [][]check{verdicts, nil} {
		result := receipt{OperationProof: overgodb.OperationProof{Checks: refused}}
		if err := conclude(&result, true, swap); err == nil || result.Passed || result.Swapped || swaps != 0 {
			t.Fatalf("refused verdict %+v: err=%v result=%+v swaps=%d", refused, err, result, swaps)
		}
	}
	clean := receipt{OperationProof: overgodb.OperationProof{Checks: []check{{Name: "only", Passed: true}}}}
	if err := conclude(&clean, false, swap); err != nil || !clean.Passed || clean.Swapped || swaps != 0 {
		t.Fatalf("clean verdict without -swap: err=%v result=%+v swaps=%d", err, clean, swaps)
	}
	if err := conclude(&clean, true, swap); err != nil || !clean.Swapped || clean.Superseded != "superseded" || swaps != 1 {
		t.Fatalf("clean verdict with -swap: err=%v result=%+v swaps=%d", err, clean, swaps)
	}
}

// TestProofCommitsUnderThisCommandsCapability binds the command's producer
// to the store's table: the proof it builds is admitted through its own
// capability and refused to the plain door, so a mismatch between the two
// names cannot wait for a real candidate to surface.
func TestProofCommitsUnderThisCommandsCapability(t *testing.T) {
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	proof := overgodb.OperationProof{Checks: []check{{Name: "only", Passed: true}}}
	proof.Verdict()
	batch, err := proof.Batch(t.Context(), store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); !errors.Is(err, overgodb.ErrProducerRefused) {
		t.Fatalf("the plain door admitted a proof: %v", err)
	}
	if _, err := store.CommitAs(t.Context(), producer, batch); err != nil {
		t.Fatalf("this command's capability was refused its own proof: %v", err)
	}
}

// TestSwapStoresKeepsTheSupersededStore renames the live store aside and the
// candidate into service, refuses an occupied superseded path, and restores
// the live store when the candidate cannot take its place.
func TestSwapStoresKeepsTheSupersededStore(t *testing.T) {
	base := t.TempDir()
	live, candidate := filepath.Join(base, "store"), filepath.Join(base, "store.candidate")
	mark := func(directory, text string) {
		t.Helper()
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, "marker"), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	read := func(directory string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(directory, "marker"))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	mark(live, "live")
	mark(candidate, "candidate")
	superseded, err := swapStores(live, candidate, "bfc908c35167e34af162")
	if err != nil || read(live) != "candidate" || read(superseded) != "live" || filepath.Base(superseded) != "store.superseded-bfc908c35167" {
		t.Fatalf("swap = (%q, %v)", superseded, err)
	}
	mark(candidate, "second")
	if _, err := swapStores(live, candidate, "bfc908c35167e34af162"); err == nil || read(live) != "candidate" {
		t.Fatalf("swap over an occupied superseded path = %v", err)
	}
	if _, err := swapStores(live, filepath.Join(base, "absent"), "0123456789abcdef"); err == nil || read(live) != "candidate" {
		t.Fatalf("swap without a candidate left the live store at %q: %v", read(live), err)
	}
}
