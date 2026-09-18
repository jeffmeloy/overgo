package dataroot

import (
	"path/filepath"
	"testing"
)

// TestStoreRoot proves the shared resolver: a non-empty -repo flag overrides
// the contract verbatim (trimmed), and an empty flag falls back to the
// resolved store root, so every command resolves the store the same way.
func TestStoreRoot(t *testing.T) {
	t.Run("flag overrides the contract", func(t *testing.T) {
		root, err := StoreRoot("  C:/somewhere/overgodb-store  ")
		if err != nil {
			t.Fatal(err)
		}
		if root != "C:/somewhere/overgodb-store" {
			t.Fatalf("flag not honored verbatim after trim: %q", root)
		}
	})

	t.Run("empty flag falls back to the contract store", func(t *testing.T) {
		dir := t.TempDir()
		t.Chdir(dir)
		t.Setenv(Env, dir)
		root, err := StoreRoot("")
		if err != nil {
			t.Fatal(err)
		}
		roots, err := ResolveCurrent()
		if err != nil {
			t.Fatal(err)
		}
		if root != roots.Store {
			t.Fatalf("fallback %q differs from the resolved store %q", root, roots.Store)
		}
		if root != filepath.Join(dir, "overgodb-store") {
			t.Fatalf("fallback store %q is not the working-directory default", root)
		}
	})
}
