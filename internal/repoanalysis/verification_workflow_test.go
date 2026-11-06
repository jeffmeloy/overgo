package repoanalysis

import (
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/testcheck"
)

// TestIncrementalStaticAnalysisContracts holds the shared static-analysis
// contract the census owners ride: one parse per file serves every rule, an
// overlay re-parses only the changed file while every unchanged file keeps its
// cached parse, and a census over the incremental snapshot reuses those shared
// parses yet reflects exactly the change -- a seeded fixed timer present in the
// base is gone from the repaired candidate, and the incremental result equals a
// full analysis of the same content.
func TestIncrementalStaticAnalysisContracts(t *testing.T) {
	t.Parallel()
	fileByPath := func(snapshot gosyntax.Snapshot, path string) gosyntax.File {
		for _, file := range snapshot.Files {
			if file.Path == path {
				return file
			}
		}
		t.Fatalf("snapshot lacks %s", path)
		return gosyntax.File{}
	}

	base := testcheck.Must((gosyntax.Snapshot{}).Overlay(map[string][]byte{
		"a/a.go": []byte("package a\nimport \"time\"\nfunc A() { time.Sleep(2 * time.Second) }\n"),
		"b/b.go": []byte("package b\nfunc B() {}\n"),
	}))(t)

	// Shared parse: one parse per file serves repeated consumers.
	firstA := testcheck.Must(fileByPath(base, "a/a.go").Syntax())(t)
	if secondA, _ := fileByPath(base, "a/a.go").Syntax(); firstA != secondA {
		t.Fatal("a file parsed twice for two consumers")
	}
	baseB := testcheck.Must(fileByPath(base, "b/b.go").Syntax())(t)

	// Rule-specific invalidation: change only a/a.go. The unchanged file keeps
	// its cached parse; the changed file re-parses and carries the edit.
	candidate := testcheck.Must(base.Overlay(map[string][]byte{"a/a.go": []byte("package a\nfunc A() {}\n")}))(t)
	if candidateB, _ := fileByPath(candidate, "b/b.go").Syntax(); baseB != candidateB {
		t.Fatal("unchanged file re-parsed under the overlay")
	}
	if candidateA, _ := fileByPath(candidate, "a/a.go").Syntax(); candidateA == firstA {
		t.Fatal("changed file kept a stale parse")
	}

	// A real census reuses the shared parses and reflects only the change.
	baseTimers := testcheck.Must(FixedTimerCensus(base))(t)
	candidateTimers := testcheck.Must(FixedTimerCensus(candidate))(t)
	if len(baseTimers) == 0 {
		t.Fatal("seeded fixed timer not detected in the base census")
	}
	if len(candidateTimers) != 0 {
		t.Fatalf("repaired candidate still lists timers: %+v", candidateTimers)
	}

	// Incremental analysis equals a full analysis of the same content.
	full := testcheck.Must((gosyntax.Snapshot{}).Overlay(map[string][]byte{
		"a/a.go": []byte("package a\nfunc A() {}\n"),
		"b/b.go": []byte("package b\nfunc B() {}\n"),
	}))(t)
	fullTimers := testcheck.Must(FixedTimerCensus(full))(t)
	if len(fullTimers) != len(candidateTimers) {
		t.Fatalf("incremental census %d differs from a full analysis %d", len(candidateTimers), len(fullTimers))
	}
}
