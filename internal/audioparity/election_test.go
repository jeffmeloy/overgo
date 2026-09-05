package audioparity

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestFirstASRArtifactElection(t *testing.T) {
	encoded, err := os.ReadFile(filepath.Join("testdata", "granite_speech_5_election.json"))
	if err != nil {
		t.Fatal(err)
	}
	election, err := NormalizeElection(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := election.ID.String(), "evidence:sha256:ea234b4a2df12eb45c6487146a8d88ec11b492ce829b4f5568aab4b2a5e33f91"; got != want {
		t.Fatalf("election identity = %s, want %s", got, want)
	}
	if got, want := election.Model.ID.String(), "model:sha256:1712f360981e18b30efad1e17611d16066549b820da88c0eb7c7b6f6ad39c098"; got != want {
		t.Fatalf("model identity = %s, want %s", got, want)
	}
	if len(election.Files) != 14 || election.License.SPDX != "Apache-2.0" || election.Runtime.Device != "cpu" {
		t.Fatalf("incomplete election: files=%d license=%s device=%s", len(election.Files), election.License.SPDX, election.Runtime.Device)
	}
	summary := election.Summary()
	if summary.Fixtures != 4 || summary.StandardMatches != 3 || summary.EquivalentMatches != 4 ||
		summary.AudioSamples != 342560 || summary.WallNS != 968816000 {
		t.Fatalf("summary = %+v", summary)
	}

	batch, err := election.Batch("audio/oracle/granite-speech-5")
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	selected, found, err := store.ResolveAlias(t.Context(), FirstOfflineASRAlias)
	if err != nil || !found || selected != election.ID {
		t.Fatalf("selected election = (%s, %t, %v)", selected, found, err)
	}
	manifest, found, err := store.Manifest(t.Context(), election.Model.ID)
	if err != nil || !found || manifest.ID != election.Model.ID {
		t.Fatalf("registered model = (%s, %t, %v)", manifest.ID, found, err)
	}
	stored, found, err := artifact.ReadContent(t.Context(), store, election.ID)
	if err != nil || !found {
		t.Fatalf("registered election = (%t, %v)", found, err)
	}
	replayed, err := electionCodec.Parse(stored.Data)
	if err != nil || replayed.ID != election.ID {
		t.Fatalf("replayed election = (%s, %v)", replayed.ID, err)
	}

	withoutReference := cloneElection(election)
	withoutReference.Files = slicesWithoutRole(withoutReference.Files, "reference-code")
	withoutReference.ID = artifact.ID{}
	if _, err := electionCodec.NewInitial(withoutReference); err == nil {
		t.Fatal("election accepted without reference execution bytes")
	}
	withoutParity := cloneElection(election)
	withoutParity.Observations[0].Observed = "different transcript"
	withoutParity.ID = artifact.ID{}
	if _, err := electionCodec.NewInitial(withoutParity); err == nil {
		t.Fatal("election accepted without pinned transcript parity")
	}
}

func slicesWithoutRole(files []FileBinding, role string) []FileBinding {
	result := files[:0]
	for _, file := range files {
		if file.Role != role {
			result = append(result, file)
		}
	}
	return result
}
