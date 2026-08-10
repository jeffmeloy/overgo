package dataroot

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEnvBaseWinsAndNamesAllRoots(t *testing.T) {
	base := t.TempDir()
	t.Setenv(Env, base)
	roots, err := Resolve(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if roots.Source != Env || roots.Models != filepath.Join(base, "models") ||
		roots.Store != filepath.Join(base, "repodb-store") ||
		roots.Checkpoints != filepath.Join(base, "checkpoints") {
		t.Fatalf("env resolution wrong: %+v", roots)
	}
}

func TestEnvPointingNowhereFailsLoudly(t *testing.T) {
	t.Setenv(Env, filepath.Join(t.TempDir(), "absent"))
	if _, err := Resolve(t.TempDir()); err == nil {
		t.Fatal("absent env base resolved silently")
	}
}

func TestConfigFileOverridesPerRootWithDefaultsForOmitted(t *testing.T) {
	t.Setenv(Env, "")
	work := t.TempDir()
	elsewhere := t.TempDir()
	config := `{"models":"` + filepath.ToSlash(filepath.Join(elsewhere, "weights")) + `"}`
	if err := os.WriteFile(filepath.Join(work, ConfigFile), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	roots, err := Resolve(work)
	if err != nil {
		t.Fatal(err)
	}
	if roots.Source != ConfigFile {
		t.Fatalf("expected config source, got %q", roots.Source)
	}
	if roots.Models != filepath.Join(elsewhere, "weights") {
		t.Fatalf("models override lost: %q", roots.Models)
	}
	if roots.Store != filepath.Join(work, "repodb-store") {
		t.Fatalf("omitted store should default to working directory: %q", roots.Store)
	}
}

func TestUnknownConfigFieldFailsLoudly(t *testing.T) {
	t.Setenv(Env, "")
	work := t.TempDir()
	if err := os.WriteFile(filepath.Join(work, ConfigFile), []byte(`{"modles":"/typo"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Resolve(work); err == nil {
		t.Fatal("unknown field accepted; a typo would silently lose the override")
	}
}

func TestDefaultsPreservePreContractBehavior(t *testing.T) {
	t.Setenv(Env, "")
	work := t.TempDir()
	roots, err := Resolve(work)
	if err != nil {
		t.Fatal(err)
	}
	if roots.Source != "defaults" || roots.Store != filepath.Join(work, "repodb-store") {
		t.Fatalf("fallback wrong: %+v", roots)
	}
}

func TestResolveModelPathPrefersExistingThenRoots(t *testing.T) {
	t.Setenv(Env, "")
	work := t.TempDir()
	roots, err := Resolve(work)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(roots.Models, "qwen"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(roots.Checkpoints, "tuned"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := roots.ResolveModelPath("qwen"); got != filepath.Join(roots.Models, "qwen") {
		t.Fatalf("bare name should resolve under models root: %q", got)
	}
	if got := roots.ResolveModelPath("tuned"); got != filepath.Join(roots.Checkpoints, "tuned") {
		t.Fatalf("checkpoint name should resolve under checkpoints root: %q", got)
	}
	if got := roots.ResolveModelPath("missing-model"); got != "missing-model" {
		t.Fatalf("unresolvable reference must return unchanged: %q", got)
	}
	existing := filepath.Join(work, "here.gguf")
	if err := os.WriteFile(existing, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := roots.ResolveModelPath(existing); got != existing {
		t.Fatalf("existing path must win unchanged: %q", got)
	}
}
