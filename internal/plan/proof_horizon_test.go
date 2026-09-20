package plan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestCompletionProofHorizon holds the authority to its horizon. A landed
// completion whose store evidence is gone refuses resolution until a
// committed receipt names it; a receipt in the working tree alone skips no
// proof; at or before the horizon the completion is accepted from Git alone
// and its identity still feeds dispatch and the duplicate ratchet; a receipt
// whose count differs from what history holds, or whose commit is not in the
// history, is refused; and the only horizon an authority seals is its own
// revision with the count it proved.
func TestCompletionProofHorizon(t *testing.T) {
	t.Parallel()
	head := func(fixture *completionFixture) string {
		return strings.TrimSpace(string(runGit(t, fixture.repository, nil, "rev-parse", "HEAD")))
	}
	write := func(fixture *completionFixture, horizon ProofHorizon, commit bool) {
		t.Helper()
		path := filepath.Join(fixture.repository, filepath.FromSlash(ProofHorizonPath))
		data, err := json.Marshal(horizon)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		if commit {
			runGit(t, fixture.repository, nil, "add", "--", ProofHorizonPath)
			runGit(t, fixture.repository, []byte("seal the proof horizon\n"), "commit", "-q", "-F", "-")
		}
	}

	proven := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
	proven.commit(proven.canonicalMessage(), true)
	authority, err := resolveFixture(proven, proven.child, "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if sealed := authority.SealProofHorizon(); sealed.Commit != head(proven) || sealed.Completions != 1 {
		t.Fatalf("sealed horizon = %+v, want this revision with its one proven completion", sealed)
	}

	// The same landing with no store evidence at all.
	bare := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
	landed := bare.commit(bare.canonicalMessage(), false)
	if _, err := resolveFixture(bare, bare.child, "HEAD"); err == nil {
		t.Fatal("a completion with no store evidence and no horizon was accepted")
	}
	write(bare, ProofHorizon{Commit: landed, Completions: 1}, false)
	if _, err := resolveFixture(bare, bare.child, "HEAD"); err == nil {
		t.Fatal("an uncommitted horizon skipped a proof")
	}
	write(bare, ProofHorizon{Commit: landed, Completions: 1}, true)
	authority, err = resolveFixture(bare, bare.child, "HEAD")
	if err != nil {
		t.Fatalf("a completion at the horizon was not accepted from Git alone: %v", err)
	}
	if !authority.completed("root/do") {
		t.Fatal("a completion accepted at the horizon left the identity ratchet")
	}
	if sealed := authority.SealProofHorizon(); sealed.Commit != head(bare) || sealed.Completions != 1 {
		t.Fatalf("resealed horizon = %+v, want the new revision carrying the count forward", sealed)
	}

	write(bare, ProofHorizon{Commit: landed, Completions: 2}, true)
	if _, err := resolveFixture(bare, bare.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "sealed 2") {
		t.Fatalf("a horizon that miscounts history = %v", err)
	}
	write(bare, ProofHorizon{Commit: head(proven), Completions: 1}, true)
	if _, err := resolveFixture(bare, bare.child, "HEAD"); err == nil || !strings.Contains(err.Error(), "not in the history") {
		t.Fatalf("a horizon outside this history = %v", err)
	}
}
