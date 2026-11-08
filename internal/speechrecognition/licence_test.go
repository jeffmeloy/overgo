package speechrecognition

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAModelWithoutALicenceIsServed holds the speaker model to the owner's rule of 2026-11-08: no model is ever
// distributed, so its licence binds only its local use and loading reads none. Its profile demanded a licence file,
// bound it in its lineage, and the model refused to load without one. A profile with no licence is made, and its
// lineage binds the model and inventory alone; one with a licence binds it too.
func TestAModelWithoutALicenceIsServed(t *testing.T) {
	t.Parallel()
	var declaration struct {
		Encoder  Declaration
		Activity ActivityBinding
	}
	testcheck.NoError(t, jsonfile.Decode("../audioparity/testdata/speaker_activity_declaration.json", &declaration))
	profile := SpeakerProfile{
		Model:         testutil.ArtifactID(t, artifact.KindModel, "unlicensed speaker model"),
		Inventory:     testutil.ArtifactID(t, artifact.KindTensorInventory, "unlicensed speaker tensors"),
		MaximumOffset: 1e-3, Encoder: declaration.Encoder, Activity: declaration.Activity,
		Boundary: SpeakerBoundary{Threshold: .5, FrameSamples: 1280, TickSamples: 160, FinalBoundary: "last-tick-clipped-to-source"},
	}
	testcheck.NoError(t, jsonfile.Decode("../audioparity/testdata/speaker_frontend.json", &profile.Frontend))
	made := testcheck.Must(NewSpeakerProfile(profile))(t)
	parents := func(profile SpeakerProfile) []artifact.ID {
		var ids []artifact.ID
		for _, lineage := range profile.Lineage() {
			ids = append(ids, lineage.Parent)
		}
		return ids
	}
	if made.License.Valid() || !slices.Equal(parents(made), []artifact.ID{made.Model, made.Inventory}) {
		t.Fatalf("a profile with no licence binds %v", parents(made))
	}
	profile.License = testutil.ArtifactID(t, artifact.KindFile, "speaker licence")
	if licensed := testcheck.Must(NewSpeakerProfile(profile))(t); !slices.Contains(parents(licensed), profile.License) {
		t.Fatalf("a profile with a licence binds %v, without it", parents(licensed))
	}
}
