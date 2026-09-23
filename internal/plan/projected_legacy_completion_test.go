package plan

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestProjectedCompletionCarriesLegacyEvidence keeps a completion that
// predates gate attempts in a merge projection: it validates and encodes with
// its attempt, result and finalization absent, while a completion with only
// some of them stays refused. A first-parent-target merge whose history
// reached such a completion once failed to encode the projection at all.
func TestProjectedCompletionCarriesLegacyEvidence(t *testing.T) {
	t.Parallel()
	contract := sha256.Sum256([]byte("legacy completion contract"))
	legacy := ProjectedCompletionEvidence{
		Reference: "legacy-item/evidence", Commit: "64b5e87a86f4f9fa78c1df3b2701ff6ae1f8adc3",
		Verify: "go build ./...", ContractDigest: hex.EncodeToString(contract[:]),
		Manifest:     testutil.ArtifactID(t, artifact.KindRecipe, "legacy-manifest"),
		CodeManifest: testutil.ArtifactID(t, artifact.KindProfile, "legacy-code-manifest"),
	}
	if err := validateProjectedCompletionEvidence(legacy); err != nil {
		t.Fatalf("legacy completion refused: %v", err)
	}
	if _, err := json.Marshal(legacy); err != nil {
		t.Fatalf("legacy completion does not encode: %v", err)
	}
	prepared := legacy
	prepared.Attempt = testutil.ArtifactID(t, artifact.KindEvidence, "attempt")
	prepared.Result = testutil.ArtifactID(t, artifact.KindEvidence, "result")
	prepared.Finalization = testutil.ArtifactID(t, artifact.KindEvidence, "finalization")
	if err := validateProjectedCompletionEvidence(prepared); err != nil {
		t.Fatalf("prepared completion refused: %v", err)
	}
	partial := legacy
	partial.Attempt = prepared.Attempt
	if err := validateProjectedCompletionEvidence(partial); err == nil {
		t.Fatal("a completion with an attempt but no result or finalization was accepted")
	}
}
