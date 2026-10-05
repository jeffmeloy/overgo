package main

import (
	"strings"
	"testing"
)

// TestManifestNamesTheExecutablesCommit holds a hand-written manifest to the
// commit that produces its evidence: only the commit this executable is built
// from is accepted, and a stale, foreign or empty one is refused with both
// named.
func TestManifestNamesTheExecutablesCommit(t *testing.T) {
	t.Parallel()
	executable := strings.Repeat("a", 40)
	if err := requireProducer(executable, executable); err != nil {
		t.Fatalf("the executable's own commit was refused: %v", err)
	}
	for name, claimed := range map[string]string{
		"a stale commit":   strings.Repeat("b", 40),
		"an empty commit":  "",
		"an abbreviation":  executable[:12],
		"a differing case": strings.ToUpper(executable),
	} {
		err := requireProducer(claimed, executable)
		if err == nil || !strings.Contains(err.Error(), executable[:12]) {
			t.Fatalf("%s was accepted as the producer or its refusal does not name the executable: %v", name, err)
		}
	}
}
