package repoanalysis

import "testing"

// TestIncrementalStaticAnalysisContracts holds the shared static-analysis
// contract the census owners ride: one parse per file serves every rule, an
// overlay re-parses only the changed file while every unchanged file keeps its
// cached parse, and a census over the incremental snapshot reuses those shared
// parses yet reflects exactly the change -- a seeded fixed timer present in the
// base is gone from the repaired candidate, and the incremental result equals a
// full analysis of the same content.
func TestIncrementalStaticAnalysisContracts(t *testing.T) {
	t.Parallel()
	fileByPath := func(snapshot SourceSnapshot, path string) GoFile {
		for _, file := range snapshot.Files {
			if file.Path == path {
				return file
			}
		}
		t.Fatalf("snapshot lacks %s", path)
		return GoFile{}
	}

	base, err := (SourceSnapshot{}).Overlay(map[string][]byte{
		"a/a.go": []byte("package a\nimport \"time\"\nfunc A() { time.Sleep(2 * time.Second) }\n"),
		"b/b.go": []byte("package b\nfunc B() {}\n"),
	})
	if err != nil {
		t.Fatal(err)
	}

	// Shared parse: one parse per file serves repeated consumers.
	firstA, err := fileByPath(base, "a/a.go").Syntax()
	if err != nil {
		t.Fatal(err)
	}
	if secondA, _ := fileByPath(base, "a/a.go").Syntax(); firstA != secondA {
		t.Fatal("a file parsed twice for two consumers")
	}
	baseB, err := fileByPath(base, "b/b.go").Syntax()
	if err != nil {
		t.Fatal(err)
	}

	// Rule-specific invalidation: change only a/a.go. The unchanged file keeps
	// its cached parse; the changed file re-parses and carries the edit.
	candidate, err := base.Overlay(map[string][]byte{"a/a.go": []byte("package a\nfunc A() {}\n")})
	if err != nil {
		t.Fatal(err)
	}
	if candidateB, _ := fileByPath(candidate, "b/b.go").Syntax(); baseB != candidateB {
		t.Fatal("unchanged file re-parsed under the overlay")
	}
	if candidateA, _ := fileByPath(candidate, "a/a.go").Syntax(); candidateA == firstA {
		t.Fatal("changed file kept a stale parse")
	}

	// A real census reuses the shared parses and reflects only the change.
	baseTimers, err := FixedTimerCensus(base)
	if err != nil {
		t.Fatal(err)
	}
	candidateTimers, err := FixedTimerCensus(candidate)
	if err != nil {
		t.Fatal(err)
	}
	if len(baseTimers) == 0 {
		t.Fatal("seeded fixed timer not detected in the base census")
	}
	if len(candidateTimers) != 0 {
		t.Fatalf("repaired candidate still lists timers: %+v", candidateTimers)
	}

	// Incremental analysis equals a full analysis of the same content.
	full, err := (SourceSnapshot{}).Overlay(map[string][]byte{
		"a/a.go": []byte("package a\nfunc A() {}\n"),
		"b/b.go": []byte("package b\nfunc B() {}\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	fullTimers, err := FixedTimerCensus(full)
	if err != nil {
		t.Fatal(err)
	}
	if len(fullTimers) != len(candidateTimers) {
		t.Fatalf("incremental census %d differs from a full analysis %d", len(candidateTimers), len(fullTimers))
	}
}
