package main

import (
	"strings"
	"testing"

	"overgo/internal/runrecord"
)

// TestHistoryRanksReaders holds the phase history to naming which unnamed
// readers a landing paid for: each reader is printed with the packages it
// selected and their summed test time, the widest first, and a package
// selected for the compiler alone adds to no reader.
func TestHistoryRanksReaders(t *testing.T) {
	t.Parallel()
	reader := func(detail string) runrecord.SelectionCause {
		return runrecord.SelectionCause{Kind: runrecord.SelectionCauseReader, Detail: detail}
	}
	wide, narrow := "overgo/internal/wide: reads a path the source does not name", "overgo/internal/narrow: reads a path the source does not name"
	histogram := runrecord.SelectionHistogram{Packages: []runrecord.SelectionPackageCauses{
		{Package: "overgo/cmd/a", Step: "test", ElapsedSeconds: new(2.0), Causes: []runrecord.SelectionCause{reader(wide), reader(narrow)}},
		{Package: "overgo/cmd/b", Step: "test", ElapsedSeconds: new(3.5), Causes: []runrecord.SelectionCause{reader(wide)}},
		{Package: "overgo/cmd/c", Step: "test", ElapsedSeconds: new(9.0), Causes: []runrecord.SelectionCause{{Kind: runrecord.SelectionCauseCompiler, Detail: "cmd/c/main.go"}}},
	}}
	var output strings.Builder
	writeSelectionCauses(&output, &histogram)
	first := strings.Index(output.String(), "reader packages=2 elapsed=5.5s "+wide)
	second := strings.Index(output.String(), "reader packages=1 elapsed=2.0s "+narrow)
	if first < 0 || second < first {
		t.Fatalf("history = %q, want the wide reader ranked before the narrow one with their packages and time", output.String())
	}
	if strings.Count(output.String(), "\n  reader ") != 2 {
		t.Fatalf("history = %q, want exactly the two readers", output.String())
	}
}
