package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
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

// TestFailedReplacementKeepsTheLivePath holds a rebuilt store to the one
// cutover precheck owns. A rebuild whose consumer checks refuse it never
// replaces the store in service, and a destination that already holds a store
// is refused before anything is written to it. The swap renames the whole
// store, so the rebuilt journal goes into service with the blobs it
// externalized; a swap whose candidate cannot take the live path restores the
// live store, and one over an occupied superseded path never starts.
func TestFailedReplacementKeepsTheLivePath(t *testing.T) {
	base := t.TempDir()
	live, candidate := filepath.Join(base, "store"), filepath.Join(base, "store.candidate")
	contents := []artifact.Content{
		textContent(t, "held inline in the journal"),
		textContent(t, strings.Repeat("held in a blob ", 1<<10)),
	}
	store, err := overgodb.Open(live)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "cutover/fixture", Contents: contents}); err != nil {
		t.Fatal(err)
	}
	original, _ := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	// serves opens the store at root and requires head and every content.
	serves := func(root string, head artifact.CommitID, extra ...artifact.Content) {
		t.Helper()
		store, err := overgodb.OpenReadOnly(root)
		if err != nil {
			t.Fatalf("%s does not open: %v", root, err)
		}
		defer store.Close()
		if got, _ := store.Head(); got != head {
			t.Fatalf("%s is at head %s, want %s", root, got, head)
		}
		for _, content := range append(contents, extra...) {
			got, found, err := artifact.ReadContent(t.Context(), store, content.Descriptor.ID)
			if err != nil || !found || !bytes.Equal(got.Data, content.Data) {
				t.Fatalf("%s does not serve %s: found=%t %v", root, content.Descriptor.ID, found, err)
			}
		}
	}
	// A checkout that holds no overgodb command fails the published-chains check.
	rebuildArgs := []string{"rebuild", "-repo", live, "-destination", candidate, "-checkout", t.TempDir(), "-swap"}
	if err := runSubcommand(rebuildArgs); err == nil || !strings.Contains(err.Error(), "nothing was swapped") {
		t.Fatalf("a refused rebuild = %v, want nothing swapped", err)
	}
	serves(live, original)
	rebuilt, err := overgodb.OpenReadOnly(candidate)
	if err != nil {
		t.Fatal(err)
	}
	head, _ := rebuilt.Head()
	if err := rebuilt.Close(); err != nil {
		t.Fatal(err)
	}
	if err := runSubcommand(rebuildArgs[:len(rebuildArgs)-1]); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("a rebuild into an existing store = %v", err)
	}
	serves(candidate, head)
	t.Run("completed writer invalidates the candidate", func(t *testing.T) {
		changing, stale := filepath.Join(base, "changing"), filepath.Join(base, "stale")
		for _, destination := range []string{changing, stale} {
			if _, err := overgodb.CloneStore(live, destination); err != nil {
				t.Fatal(err)
			}
		}
		writer, err := overgodb.Open(changing)
		if err != nil {
			t.Fatal(err)
		}
		tail := textContent(t, "acknowledged after the candidate snapshot")
		newer, err := writer.Commit(t.Context(), artifact.Batch{Key: "cutover/newer", Contents: []artifact.Content{tail}})
		if err := errors.Join(err, writer.Close()); err != nil {
			t.Fatal(err)
		}
		if _, err := overgodb.ReplaceStore(t.Context(), changing, stale, original.String()); err == nil {
			t.Fatal("a candidate built before an acknowledged write replaced the live store")
		}
		serves(changing, newer, tail)
		serves(stale, original)
	})
	if _, err := overgodb.ReplaceStore(t.Context(), live, filepath.Join(base, "absent"), original.String()); err == nil {
		t.Fatal("a swap without a candidate succeeded")
	}
	serves(live, original)
	superseded, err := overgodb.ReplaceStore(t.Context(), live, candidate, original.String())
	if err != nil || filepath.Base(superseded) != "store.superseded-"+original.String()[:12] {
		t.Fatalf("swap = (%q, %v)", superseded, err)
	}
	serves(live, head)
	serves(superseded, original)
	if err := os.Mkdir(candidate, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := overgodb.ReplaceStore(t.Context(), live, candidate, original.String()); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("swap over an occupied superseded path = %v", err)
	}
	serves(live, head)
}

// textContent is a plain-text evidence content holding text.
func textContent(t *testing.T, text string) artifact.Content {
	t.Helper()
	data := []byte(text)
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, data)
	if err != nil {
		t.Fatal(err)
	}
	return artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: "text/plain"}, Data: data}
}
