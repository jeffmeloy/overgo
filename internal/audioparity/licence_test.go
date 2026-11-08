package audioparity

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAModelWithoutALicenceIsServed holds the audio reference election to the owner's rule of 2026-11-08: no model is
// ever distributed, so the reference model's licence binds only its local use. The election demanded Apache-2.0 and a
// licence URL bound to a file. An election with no licence normalizes, so does one whose bound licence is MIT, and one
// naming a licence document no file binds is still refused.
func TestAModelWithoutALicenceIsServed(t *testing.T) {
	t.Parallel()
	normalized := func(election Election) error {
		data := testcheck.Must(json.Marshal(election))(t)
		_, err := NormalizeElection(data)
		return err
	}
	election := qualificationElection(t)
	licensed := election.License
	election.License = LicenseBinding{}
	testcheck.NoError(t, normalized(election))
	election.License = LicenseBinding{SPDX: "MIT", Document: licensed.Document}
	testcheck.NoError(t, normalized(election))
	election.License.Document = testutil.ArtifactID(t, artifact.KindFile, "an unbound licence")
	if err := normalized(election); err == nil {
		t.Fatal("an election naming a licence document no file binds was normalized")
	}
}
