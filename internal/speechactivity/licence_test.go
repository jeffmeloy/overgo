package speechactivity

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAModelWithoutALicenceIsServed holds the speech activity detector to the owner's rule of 2026-11-08: no model is
// ever distributed, so a model's licence binds only its local use and the detector loads whatever its licence. Its
// profile demanded a licence file and the detector refused to load without one. A profile with no licence is made and
// binds none, and a licence that is no file is still refused.
func TestAModelWithoutALicenceIsServed(t *testing.T) {
	t.Parallel()
	profile := Profile{
		Version:   artifact.InitialDocumentVersion,
		Model:     testutil.ArtifactID(t, artifact.KindModel, "unlicensed detector"),
		Inventory: testutil.ArtifactID(t, artifact.KindTensorInventory, "unlicensed detector tensors"),
		Network:   Declaration{Blocks: []MemoryBlock{{}}, Dense: []Affine{{}}},
		Offline:   &OfflineConfig{},
	}
	if made := testcheck.Must(profileCodec.New(profile))(t); made.License.Valid() || !made.ID.Valid() {
		t.Fatalf("a profile with no licence was made as %+v", made)
	}
	profile.License = testutil.ArtifactID(t, artifact.KindModel, "no licence file")
	if _, err := profileCodec.New(profile); err == nil {
		t.Fatal("a licence that is no file was recorded")
	}
}
