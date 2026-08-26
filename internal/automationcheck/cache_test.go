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
	planned, err := Plan([]Check{check}, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	cache := NewEvidenceCache(environment)
	first, reused, err := cache.RunCached(context.Background(), planned[0], input)
	if err != nil || reused || runs != 1 {
		t.Fatalf("first = (%+v, %t, %v), runs=%d", first, reused, err, runs)
	}
	second, reused, err := cache.RunCached(context.Background(), planned[0], input)
	if err != nil || !reused || runs != 1 || second.ID != first.ID || !second.Reused || second.Inapplicable {
		t.Fatalf("second = (%+v, %t, %v), runs=%d", second, reused, err, runs)
	}
	otherEnvironment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("other"))
	if cache.Reusable(otherEnvironment) {
		t.Fatal("cache crossed environment identity")
	}
}

func TestCacheBoundedEviction(t *testing.T) {
	environment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("environment"))
	check := Check{
		Descriptor: Descriptor{Name: "bounded", Phase: runrecord.PhaseTest, Always: true},
		Run:        func(context.Context, Invocation) (bool, string, error) { return false, "", nil },
	}
	planned, err := Plan([]Check{check}, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	cache := NewEvidenceCache(environment)
	for index := range 8 {
		input, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte{byte(index + 1)})
		if err != nil {
			t.Fatal(err)
		}
		if _, reused, err := cache.RunCached(context.Background(), planned[0], input); err != nil || reused {
			t.Fatalf("input %d = reused %t, %v", index, reused, err)
		}
		if len(cache.Entries) != 1 {
			t.Fatalf("cache grew to %d entries for one invocation", len(cache.Entries))
		}
	}
	legacyInput, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("legacy"))
	cache.Entries[planned[0].ID.String()+"\x00"+legacyInput.String()] = CacheEntry{Invocation: planned[0].ID, Input: legacyInput}
	cache.Compact()
	if len(cache.Entries) != 1 {
		t.Fatalf("legacy input-keyed entry survived compaction: %+v", cache.Entries)
	}
}
