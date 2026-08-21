package pathidentity

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFilesystemIdentityContract(t *testing.T) {
	root := t.TempDir()
	child := filepath.Join(root, "models")
	if err := os.Mkdir(child, 0o755); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(child, "weights.gguf")
	if err := os.WriteFile(file, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	canonical, err := Canonical(filepath.Join(root, ".", "models", "weights.gguf"))
	if err != nil {
		t.Fatal(err)
	}
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
