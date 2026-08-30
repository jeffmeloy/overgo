package automationcheck

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// TestGateReusesUnchangedCheckEvidence pins the reuse substrate the gate
// rides across consecutive attempts: identical invocation and input reuse
// the recorded terminal evidence without executing, a differing input always
// executes, and a failed run never enters the cache, so reuse can never
// launder a failure into a pass.
func TestGateReusesUnchangedCheckEvidence(t *testing.T) {
	environment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("attempt environment"))
	unchanged, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("unchanged inputs"))
	repaired, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("repaired inputs"))
	runs, failing := 0, true
	check := Check{
		Descriptor: Descriptor{Name: "attempt", Phase: runrecord.PhaseValidate, Always: true},
		Run: func(context.Context, Invocation) (bool, string, error) {
			runs++
			if failing {
				return false, "", errors.New("refused")
			}
			return false, "", nil
		},
	}
	planned, err := Plan([]Check{check}, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	cache := NewEvidenceCache(environment)
	if _, reused, err := cache.RunCached(context.Background(), planned[0], unchanged); err == nil || reused {
		t.Fatalf("failing attempt = (reused=%t, %v)", reused, err)
	}
	if len(cache.Entries) != 0 {
		t.Fatal("a failed run entered the reuse cache")
	}
	failing = false
	first, reused, err := cache.RunCached(context.Background(), planned[0], unchanged)
	if err != nil || reused || runs != 2 {
		t.Fatalf("first passing attempt = (%+v, %t, %v) runs=%d", first, reused, err, runs)
	}
	second, reused, err := cache.RunCached(context.Background(), planned[0], unchanged)
	if err != nil || !reused || runs != 2 || second.ID != first.ID || !second.Reused {
		t.Fatalf("unchanged retry = (%+v, %t, %v) runs=%d", second, reused, err, runs)
	}
	third, reused, err := cache.RunCached(context.Background(), planned[0], repaired)
	if err != nil || reused || runs != 3 || third.Reused {
		t.Fatalf("changed-input retry = (%+v, %t, %v) runs=%d", third, reused, err, runs)
	}
}
