package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

// TestBindingSurvivesANewUse holds the ledger to keeping a reviewed decision
// when only the uses of its constant move. A constant is triaged, then gains
// a use in another function, which drifts its callsite identity and nothing
// else. Retiring the unmatched decision is refused by name, because it is not
// stale and retiring it would throw its understanding away; the reviewed
// import rebinds it, and the decision that is active afterwards carries the
// understanding it was triaged with.
func TestBindingSurvivesANewUse(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "sample", "policy.go")
	write := func(source string) repoanalysis.SourceSnapshot {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
		snapshot, err := repoanalysis.DiscoverGo(root, "internal")
		if err != nil {
			t.Fatal(err)
		}
		return snapshot
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	const declared = "package sample\n\nconst PolicyWindow = 3 * 8\n\nfunc First() int { return PolicyWindow }\n"
	write(declared)
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	var candidate closurescan.Candidate
	for _, scanned := range candidates {
		if scanned.Name == "PolicyWindow" {
			candidate = scanned
		}
	}
	const understanding = "Fixed fixture policy, reviewed once."
	encoded, err := json.Marshal(triageFile{Rows: []triageRow{{
		Kind: candidate.Kind, Name: candidate.Name, File: candidate.File, Scope: candidate.Scope, Line: candidate.Line,
		Tier: string(closureledger.TierImplementation), Status: string(closureledger.StatusClosed),
		Understanding: understanding, ClosurePath: "Replace when the fixture contract changes.",
		RerankTrigger: "Fixture contract change.",
	}}})
	if err != nil {
		t.Fatal(err)
	}
	triagePath := filepath.Join(root, "triage.json")
	if err := os.WriteFile(triagePath, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := emit(root, "store", triagePath, candidates); err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(root, "store")

	used := write(declared + "\nfunc Second() int { return PolicyWindow }\n")
	if _, unmatched, first, _, err := importClosureDocuments(root, "store", storePath, used, false, false); err != nil || unmatched != 1 || first != "PolicyWindow:callsite" {
		t.Fatalf("a new use = (unmatched=%d first=%s, %v), want the decision drifted at its callsites", unmatched, first, err)
	}
	if _, _, _, _, err := importClosureDocuments(root, "store", storePath, used, false, true); err == nil || !strings.Contains(err.Error(), "-review-callsites") {
		t.Fatalf("retiring a decision that drifted only at its callsites = %v, want it refused with the flag that keeps it", err)
	}
	if count, unmatched, first, _, err := importClosureDocuments(root, "store", storePath, used, true, false); err != nil || count != 1 || unmatched != 0 {
		t.Fatalf("the reviewed import = (rebound=%d unmatched=%d first=%s, %v), want the decision rebound", count, unmatched, first, err)
	}

	rescanned, err := closurescan.ScanSnapshot(used, nil, closurescan.CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, current := range rescanned {
		if current.Name != "PolicyWindow" {
			continue
		}
		binding, err := current.Binding()
		if err != nil {
			t.Fatal(err)
		}
		document, active, err := closureledger.ResolveActiveBinding(t.Context(), store, binding, current.ValueJSON())
		if err != nil || !active || document.Understanding != understanding {
			t.Fatalf("after the reviewed import the decision = (%q, active=%t, %v), want the understanding it was triaged with", document.Understanding, active, err)
		}
	}
}
