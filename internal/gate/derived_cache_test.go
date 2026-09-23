package gate

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDerivedCacheStoredOnlyWhenReused holds the gate to storing a derived
// cache only when a later gate reads it back. Measured 2026-09-23: the
// modern-Go memo is keyed to a whole source snapshot and read back as the
// next gate's base, so it stays; the code profile was written into every
// gate's record -- 218 MB held -- and nothing ever read it, since each gate
// measures its own candidate and base. So no production code writes the code
// profile, its package defines no document for it, and its schema is named
// only by the store's release list, which still finds the records held.
func TestDerivedCacheStoredOnlyWhenReused(t *testing.T) {
	t.Parallel()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	for _, top := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, top), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			source, relative := string(data), filepath.ToSlash(strings.TrimPrefix(path, root+string(filepath.Separator)))
			switch {
			case strings.Contains(source, "codeprofile.EvidenceSchema") && !strings.HasPrefix(relative, "internal/storepolicy/"):
				t.Errorf("%s names the code-profile schema outside the store's release list", relative)
			case strings.HasPrefix(relative, "internal/codeprofile/") && strings.Contains(source, "JSONDocumentCodec"):
				t.Errorf("%s defines a stored document for the code profile, which nothing reads", relative)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}
