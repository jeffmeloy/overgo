package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

// TestMovedLiteralKeepsItsCatalogEntry holds the ledger to keeping a reviewed
// decision when its declaration moves to another file of the same package
// unchanged: one import that reviews callsites and retires the unmatched
// rebinds it with the understanding it was triaged with, where it was once
// retired and had to be catalogued again by hand. A literal whose value
// changed on the way is not rebound.
func TestMovedLiteralKeepsItsCatalogEntry(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	write := func(name, source string) {
		t.Helper()
		path := filepath.Join(root, "internal", "sample", name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	const moved = "func Window() int { return 37 }\n"
	const changed = "func Stride() int { return 41 }\n"
	write("policy.go", "package sample\n\n"+moved+"\n"+changed)
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateLiterals)
	if err != nil {
		t.Fatal(err)
	}
	const understanding = "Fixed fixture window, reviewed once."
	var rows []triageRow
	for _, candidate := range candidates {
		rows = append(rows, triageRow{
			Kind: candidate.Kind, Name: candidate.Name, File: candidate.File, Scope: candidate.Scope, Line: candidate.Line,
			Tier: string(closureledger.TierImplementation), Status: string(closureledger.StatusClosed),
			Understanding: understanding + " " + candidate.Scope, ClosurePath: "Replace when the fixture contract changes.",
			RerankTrigger: "Fixture contract change.",
		})
	}
	if len(rows) == 0 {
		t.Fatal("the fixture scanned no literal")
	}
	encoded, err := json.Marshal(triageFile{Rows: rows})
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

	write("policy.go", "package sample\n\nfunc Stride() int { return 43 }\n")
	write("window.go", "package sample\n\n"+moved)
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, unmatched, first, _, err := settleClosureDocuments(root, "store", storePath, snapshot, true, true); err != nil || unmatched == 0 {
		t.Fatalf("the import = (unmatched=%d first=%s, %v), want the changed literal's decision unmatched and retired", unmatched, first, err)
	}
	rescanned, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateLiterals)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	kept := 0
	for _, current := range rescanned {
		binding, err := current.Binding()
		if err != nil {
			t.Fatal(err)
		}
		document, active, err := closureledger.ResolveActiveBinding(t.Context(), store, binding, current.ValueJSON())
		if err != nil {
			t.Fatal(err)
		}
		switch current.Scope {
		case "Window":
			if !active || document.Understanding != understanding+" Window" {
				t.Fatalf("the moved literal's decision = (%q, active=%t), want the understanding it was triaged with", document.Understanding, active)
			}
			kept++
		case "Stride":
			if active {
				t.Fatalf("a changed literal kept the decision %q", document.Understanding)
			}
		}
	}
	if kept == 0 {
		t.Fatal("the moved literal was not rescanned")
	}
}
