package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
)

// TestGateDerivesLaneEnvironment pins the lane environment: a checkout whose
// local-models.json declares another checkout's audio reference store gets
// that store exported to its tests without any operator environment, an
// operator's override still wins, and the environment evidence identifies
// the reference the acceptances read.
func TestGateDerivesLaneEnvironment(t *testing.T) {
	repo := t.TempDir()
	reference := filepath.Join(t.TempDir(), "overgodb-store")
	declaration := `{"datasets": "` + filepath.ToSlash(filepath.Join(t.TempDir(), "datasets")) + `", "audio_reference_store": "` + filepath.ToSlash(reference) + `"}`
	if err := os.WriteFile(filepath.Join(repo, dataroot.ConfigFile), []byte(declaration), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(dataroot.Env, "")
	t.Setenv(audioReferenceEnv, "")
	g := &gateContext{repo: repo}
	environment, err := g.sourceEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(environment, audioReferenceEnv+"="+reference) || !slices.Contains(environment, dataroot.Env+"="+repo) {
		t.Fatalf("lane environment = %q, want the declared reference store and the repository data root", environment)
	}
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		t.Fatal(err)
	}
	declared, err := discoverEnvironment(repo)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(audioReferenceEnv, filepath.Join(t.TempDir(), "override-store"))
	if got := audioReferenceStore(roots); got != os.Getenv(audioReferenceEnv) {
		t.Fatalf("operator override lost: %q", got)
	}
	environment, err = g.sourceEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	override := audioReferenceEnv + "=" + os.Getenv(audioReferenceEnv)
	if !slices.Contains(environment, override) || slices.Contains(environment, audioReferenceEnv+"="+reference) {
		t.Fatalf("an operator's override did not win: %q", slices.DeleteFunc(slices.Clone(environment), func(entry string) bool { return !strings.HasPrefix(entry, audioReferenceEnv) }))
	}
	overridden, err := discoverEnvironment(repo)
	if err != nil {
		t.Fatal(err)
	}
	if declared.ID == overridden.ID {
		t.Fatal("environment evidence does not identify which reference store the acceptances read")
	}
	own := t.TempDir()
	plain, err := dataroot.Resolve(own)
	if err != nil || plain.AudioReference != plain.Store {
		t.Fatalf("undeclared reference store = %q, want the checkout's own %q (%v)", plain.AudioReference, plain.Store, err)
	}
}
