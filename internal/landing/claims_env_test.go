package landing

import (
	"slices"
	"testing"

	"overgo/internal/testskip"
)

// TestTheClaimsAreReboundWithTheStoreAcceptance holds the claims generator a landing runs to the store acceptance the
// audio lane opts into (cycle-lanes/claims-env): a claim's verify is its test as named, and a claim whose test reads
// the recorded store skipped when rebound without it, keeping its old binding and refusing the change that touched its
// files (audio-cpu-heldout-resource-fitness, 2026-11-14).
func TestTheClaimsAreReboundWithTheStoreAcceptance(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"cmd/validate/main.go": "package main\n\nfunc main() {}\n"})
	l := fixture.candidate()
	at := slices.IndexFunc(l.generators(), func(held generator) bool { return slices.Contains(held.writes, "compatibility.json") })
	if at < 0 || !slices.Contains(l.generators()[at].environment, testskip.StoreAcceptanceEnv+"=1") {
		t.Fatalf("the claims generator runs in an environment without %s=1; want the store acceptance opted into", testskip.StoreAcceptanceEnv)
	}
}
