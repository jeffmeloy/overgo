package dataroot

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestEnvBaseHonoursItsOwnConfig pins the base directory named by the
// environment as a working directory of its own: its local-models.json
// redirects the bulk roots while the store stays beside it, so a tree
// elsewhere resolves the base's data exactly as the base itself does.
func TestEnvBaseHonoursItsOwnConfig(t *testing.T) {
	base := t.TempDir()
	models := filepath.Join(t.TempDir(), "vendor-models")
	testcheck.NoError(t, os.WriteFile(filepath.Join(base, ConfigFile), []byte(`{"models": "`+filepath.ToSlash(models)+`"}`), 0o600))
	t.Setenv(Env, base)
	roots, err := Resolve(t.TempDir())
	testcheck.NoError(t, err)
	if roots.Source != Env || roots.Models != models || roots.Store != filepath.Join(base, "overgodb-store") || roots.Checkpoints != filepath.Join(base, "checkpoints") {
		t.Fatalf("roots = %+v", roots)
	}

	// A base without a config keeps the default layout under itself.
	bare := t.TempDir()
	t.Setenv(Env, bare)
	roots, err = Resolve(t.TempDir())
	if err != nil || roots.Models != filepath.Join(bare, "models") || roots.Source != Env {
		t.Fatalf("bare base roots = %+v, %v", roots, err)
	}
}
