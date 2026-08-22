package automationcheck

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

func TestReusedOutcomeIsDistinctFromSkipped(t *testing.T) {
	environment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("environment"))
	input, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("inputs"))
	runs := 0
	check := Check{
		Descriptor: Descriptor{Name: "cached", Phase: runrecord.PhaseTest, Always: true},
		Run:        func(context.Context, Invocation) (bool, string, error) { runs++; return false, "", nil },
	}
	planned, err := Plan([]Check{check}, nil)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewEvidenceCache(environment)
	first, reused, err := cache.RunCached(context.Background(), planned[0], input)
	if err != nil || reused || runs != 1 {
		t.Fatalf("first = (%+v, %t, %v), runs=%d", first, reused, err, runs)
	}
	second, reused, err := cache.RunCached(context.Background(), planned[0], input)
	if err != nil || !reused || runs != 1 || second.ID != first.ID || !second.Reused || second.Skipped {
		t.Fatalf("second = (%+v, %t, %v), runs=%d", second, reused, err, runs)
	}
	otherEnvironment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("other"))
	if cache.Reusable(otherEnvironment) {
		t.Fatal("cache crossed environment identity")
	}
}
