package latentimage

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testutil"
)

// kreaModelDir: the real Krea-2-Turbo artifact under the model roots
// (READ-ONLY). Absent -> skip (worktrees / non-model hosts have no models/);
// this is a structural oracle, not a unit test of pure logic.
func kreaModelDir(t testing.TB) string {
	t.Helper()
	return testutil.ModelArtifactDir(t, "Krea-2-Turbo")
}

func kreaDirOrSkip(t *testing.T) string {
	t.Helper()
	directory := kreaModelDir(t)
	if _, err := os.Stat(filepath.Join(directory, "model_index.json")); err != nil {
		t.Skipf("UNAVAILABLE: Krea-2-Turbo not present: %v", err)
	}
	return directory
}

func kreaProfileOrSkip(t *testing.T) Profile {
	t.Helper()
	profile, err := ResolveProfile(kreaDirOrSkip(t))
	if err != nil {
		t.Fatalf("ResolveProfile: %v", err)
	}
	return profile
}

// TestVerifyCheckpoint: the CPU structural oracle -- every derived dim asserted
// against the real tensor shapes (safetensors headers only, no payload, no
// forward), and the cross-checks agree. Fails on any single disagreement.
func TestVerifyCheckpoint(t *testing.T) {
	requireLongTest(t)
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	witness, err := spec.VerifyCheckpoint(dir)
	if witness != nil {
		for _, c := range witness.SortedChecks() {
			t.Log(c.String())
		}
	}
	if err != nil {
		t.Fatalf("VerifyCheckpoint: %v", err)
	}
	t.Logf("STRUCTURAL ORACLE PASS: %d dims cross-checked vs real checkpoint (mod_fields derived=%d)",
		len(witness.Checks), witness.ModField)
}
