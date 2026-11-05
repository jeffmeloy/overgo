package modeldevice

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestExpertProfileCountsExpertsNeverRouted ranks a profile from routes some experts never took: it counts the pairs
// any run routed, which lead its ranking, and the never-routed pairs follow them, the census a pinned load is weighed
// against.
func TestExpertProfileCountsExpertsNeverRouted(t *testing.T) {
	t.Parallel()
	counts := [][]uint64{{4, 0, 1, 0}, {0, 0, 9, 2}}
	profile, err := rankExpertsByRoutes(2, 4, func(layer, expert uint32) uint64 { return counts[layer][expert] })
	testcheck.NoError(t, err)
	if profile.Routed != 4 || len(profile.Pairs) != 8 {
		t.Fatalf("profile counts %d of %d pairs routed, want 4 of 8", profile.Routed, len(profile.Pairs))
	}
	for rank, pair := range profile.Pairs {
		if routed := counts[pair.Layer][pair.Expert] != 0; routed != (rank < profile.Routed) {
			t.Fatalf("pair %d (layer %d expert %d) routed=%v ranks on the wrong side of the census", rank, pair.Layer, pair.Expert, routed)
		}
	}
	silent, err := rankExpertsByRoutes(2, 4, func(uint32, uint32) uint64 { return 0 })
	if err != nil || silent.Routed != 0 {
		t.Fatalf("a profile no run routed counts %d routed, %v", silent.Routed, err)
	}
}
