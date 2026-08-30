package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

// TestTriageProposalCarriesExactCoordinates pins the grunt-automation
// contract: a named uncatalogued candidate yields a ready-to-apply triage
// row carrying the scanner's own exact kind, file, scope, and line — the
// coordinates previously hand-copied — with a recoverable candidate's prior
// decision text prefilled and a genuinely new candidate leaving only the
// review text empty. An unknown name refuses with the current candidates
// instead of emitting a stale row.
func TestTriageProposalCarriesExactCoordinates(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "policy", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package policy\n\nconst Recovered = 21\nconst Fresh = 23\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 2 {
		t.Fatalf("candidates = (%d, %v)", len(candidates), err)
	}
	byName := map[string]closurescan.Candidate{}
	for _, candidate := range candidates {
		byName[candidate.Name] = candidate
	}
	binding, err := byName["Recovered"].Binding()
	if err != nil {
		t.Fatal(err)
	}
	binding.CallsiteID = strings.Repeat("0", 64)
	prior, err := closureledger.New(
		"Recovered", byName["Recovered"].ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
		"Recovered fixture policy.", []closureledger.SourceBinding{binding},
		"Keep the fixture policy owned.", "Fixture contract change.", binding.Owner,
	)
	if err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(root, "store")
	if _, _, err := commitClosureDocuments(
		root, storePath, closurePublishOperation, []closureledger.Document{prior}, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}

	proposal, err := proposeTriageRows(context.Background(), snapshot, store, []string{"Recovered", "Fresh"})
	if err != nil {
		t.Fatal(err)
	}
	if len(proposal.Rows) != 2 {
		t.Fatalf("proposal = %+v", proposal)
	}
	recovered, fresh := proposal.Rows[0], proposal.Rows[1]
	if recovered.Name != "Recovered" || recovered.File != byName["Recovered"].File ||
		recovered.Scope != byName["Recovered"].Scope || recovered.Line != byName["Recovered"].Line {
		t.Fatalf("recovered coordinates = %+v, scanner saw %+v", recovered, byName["Recovered"])
	}
	if recovered.Understanding != "Recovered fixture policy." || recovered.Status != string(closureledger.StatusClosed) {
		t.Fatalf("recovered row lost its prior decision text: %+v", recovered)
	}
	if fresh.Name != "Fresh" || fresh.File != byName["Fresh"].File || fresh.Line != byName["Fresh"].Line ||
		fresh.Understanding != "" {
		t.Fatalf("fresh row = %+v, want exact coordinates with empty review text", fresh)
	}

	if _, err := proposeTriageRows(context.Background(), snapshot, store, []string{"Ghost"}); err == nil ||
		!strings.Contains(err.Error(), `candidate "Ghost" not found by scan`) ||
		!strings.Contains(err.Error(), "Fresh") {
		t.Fatalf("unknown candidate emitted a stale row: %v", err)
	}
}
