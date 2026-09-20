package storepolicy

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

// TestDerivedCacheCoversModernGoComputation holds the gate's modern-Go memo
// in the releasable classes, where the record census found it missing with
// 552 MB held, and holds every class clear of the gate's own evidence kinds:
// a cache is recomputable, and nothing under the gate's prefix is.
func TestDerivedCacheCoversModernGoComputation(t *testing.T) {
	if !DerivedCache(artifact.Descriptor{Schema: repoanalysis.ModernGoComputationSchema}) {
		t.Fatalf("%s is not a releasable derived cache", repoanalysis.ModernGoComputationSchema)
	}
	for _, schema := range DerivedCacheSchemas {
		if strings.HasPrefix(schema, "overgo/gate-") {
			t.Fatalf("%s is gate evidence and is classed as a derived cache", schema)
		}
	}
}

// TestDerivedCacheNamesOnlyRecomputableSchemas holds the class list to the
// three owner-ruled schemas and the modern-Go memo, and refuses the evidence
// schemas a release must never touch.
func TestDerivedCacheNamesOnlyRecomputableSchemas(t *testing.T) {
	if len(DerivedCacheSchemas) != len([]string{"manifest", "profile", "analysis", "modern-go memo"}) {
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
