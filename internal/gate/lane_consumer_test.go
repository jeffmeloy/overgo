package gate

import (
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/codeprofile"
)

// TestLaneGateRefusesUnconsumedSurface holds a lane to master's rule that an
// export arrives with its consumer: master may stage new unconsumed surface a
// declaration names, a lane gate may not, and a merge that adds a staged
// declaration loosens master's staged-surface ratchet.
func TestLaneGateRefusesUnconsumedSurface(t *testing.T) {
	t.Parallel()
	export := codeprofile.ConsumerDeclaration{Package: "overgo/internal/vae", Name: "DecodeLatentVideo", Kind: "func", Exported: true}
	staged := codeprofile.StagedSurfaceDeclaration{Version: 2, Staged: []codeprofile.StagedSurfaceEntry{
		{Package: export.Package, Name: export.Name, Reason: "consumer lands next", RetireWith: "vae-consumer/do"},
	}}
	unconsumed := []codeprofile.ConsumerDeclaration{export}
	if accepted, blocking := admitStagedSurface(unconsumed, staged, false); len(accepted) != 1 || len(blocking) != 0 {
		t.Fatalf("master staged surface = accepted %v blocking %v", accepted, blocking)
	}
	if accepted, blocking := admitStagedSurface(unconsumed, staged, true); len(accepted) != 0 || !slices.Equal(blocking, unconsumed) {
		t.Fatalf("lane staged surface = accepted %v blocking %v, want every new export blocked", accepted, blocking)
	}
	var before, after any
	if err := json.Unmarshal([]byte(`{"staged": []}`), &before); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(staged)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &after); err != nil {
		t.Fatal(err)
	}
	if !masterRatchets["docs/staged_surface.json"] || len(looserRatchet("docs/staged_surface.json", before, after, true)) == 0 {
		t.Fatal("a merge that stages new surface does not loosen master's staged-surface ratchet")
	}
}
