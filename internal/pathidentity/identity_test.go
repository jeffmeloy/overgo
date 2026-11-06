package pathidentity

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestFilesystemIdentityContract(t *testing.T) {
	root := t.TempDir()
	child := filepath.Join(root, "models")
	testcheck.NoError(t, os.Mkdir(child, 0o755))
	file := filepath.Join(child, "weights.gguf")
	testcheck.NoError(t, os.WriteFile(file, []byte("weights"), 0o600))
	canonical := testcheck.Must(Canonical(filepath.Join(root, ".", "models", "weights.gguf")))(t)
	same, err := Same(file, canonical)
	if err != nil || !same {
		t.Fatalf("same file = (%t, %v)", same, err)
	}
	for _, candidate := range []string{root, child, file, filepath.Join(child, "future", "shard.gguf")} {
		contained, err := Contains(root, candidate)
		if err != nil || !contained {
			t.Fatalf("contained %q = (%t, %v)", candidate, contained, err)
		}
	}
	outside := filepath.Join(filepath.Dir(root), "outside.gguf")
	if contained, err := Contains(root, outside); err != nil || contained {
		t.Fatalf("outside containment = (%t, %v)", contained, err)
	}
	if _, err := Canonical(filepath.Join(root, "missing.gguf")); err == nil {
		t.Fatal("missing path received a canonical live identity")
	}

	alias := filepath.Join(root, "model-alias")
	if err := os.Symlink(child, alias); err == nil {
		aliased := filepath.Join(alias, "weights.gguf")
		if same, err := Same(file, aliased); err != nil || !same {
			t.Fatalf("linked same file = (%t, %v)", same, err)
		}
		if contained, err := Contains(root, aliased); err != nil || !contained {
			t.Fatalf("linked containment = (%t, %v)", contained, err)
		}
		outsideRoot := t.TempDir()
		escape := filepath.Join(root, "escape")
		if err := os.Symlink(outsideRoot, escape); err == nil {
			if contained, err := Contains(root, filepath.Join(escape, "future.gguf")); err != nil || contained {
				t.Fatalf("linked escape containment = (%t, %v)", contained, err)
			}
		}
	}
}
