package storepolicy

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/codeprofile"
	"overgo/internal/runrecord"
)

// TestDerivedCacheNamesOnlyRecomputableSchemas holds the class list to
// three owner-ruled schemas and refuses the evidence schemas a release
// must never touch.
func TestDerivedCacheNamesOnlyRecomputableSchemas(t *testing.T) {
	if len(DerivedCacheSchemas) != 3 {
		t.Fatalf("derived cache schemas = %v", DerivedCacheSchemas)
	}
	for _, schema := range DerivedCacheSchemas {
		if !DerivedCache(artifact.Descriptor{Schema: schema}) {
			t.Fatalf("%s is not classed as a derived cache", schema)
		}
	}
	for _, schema := range []string{"", runrecord.AttemptSchema, runrecord.GateSchema, runrecord.GateLifecycleSchema} {
		if DerivedCache(artifact.Descriptor{Schema: schema}) {
			t.Fatalf("%q is classed as a derived cache", schema)
		}
	}
}

// TestFollowChildrenNamesTheGateLifecycleChain follows children from gate
// lifecycles and gate results only; an attempt's or a cache's children are
// not part of any downward discovery.
func TestFollowChildrenNamesTheGateLifecycleChain(t *testing.T) {
	for _, schema := range []string{runrecord.GateLifecycleSchema, runrecord.GateSchema} {
		if !FollowChildren(artifact.Descriptor{Schema: schema}) {
			t.Fatalf("children of %s are not followed", schema)
		}
	}
	for _, schema := range []string{"", runrecord.AttemptSchema, codeprofile.EvidenceSchema} {
		if FollowChildren(artifact.Descriptor{Schema: schema}) {
			t.Fatalf("children of %q are followed", schema)
		}
	}
}
