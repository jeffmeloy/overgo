package landing

import (
	"slices"
	"testing"

	"overgo/internal/testutil"
)

// TestTheFixStageRunsNoClaimTest holds the fix stage's generators to commands that run no test, in the environment the
// landing has (single-source/claim-record): the claims generator ran validate -claims -run, which ran again the test
// of every claim whose source moved under the card's switch and the store acceptance, and 4 of the 21 landings refused
// on 2026-11-14 were refused there, none for wrong product behaviour. A claim's test runs where any test runs, and its
// last pass is kept; the generator writes the matrices alone, and compatibility.json is no file a generator writes.
func TestTheFixStageRunsNoClaimTest(t *testing.T) {
	t.Parallel()
	generators := (&landing{Options: Options{Repo: testutil.RepoRoot(t)}}).generators()
	claims := slices.IndexFunc(generators, func(held generator) bool { return slices.Contains(held.writes, "docs/COMPATIBILITY.md") })
	if claims < 0 {
		t.Fatal("the repository's claims generator is not among the fix stage's")
	}
	for _, held := range generators {
		if slices.Contains(held.arguments, "-run") || held.environment != nil || slices.Contains(held.writes, "compatibility.json") {
			t.Errorf("the fix stage runs %v in %d variables of its own, writing %v; want no test run, the landing's environment, and compatibility.json left to the hand that declares it",
				held.arguments, len(held.environment), held.writes)
		}
	}
}
