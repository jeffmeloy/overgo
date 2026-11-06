package repoanalysis

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestRepositoryStateParsesDirtyStatus(t *testing.T) {
	t.Parallel()
	raw := []byte(" M docs/a file.md\x00R  new.go\x00old.go\x00?? new.txt\x00")
	paths := testcheck.Must(ParseDirtyStatus(raw))(t)
	if len(paths) != 3 {
		t.Fatalf("dirty paths = %+v", paths)
	}
	var renamed DirtyPath
	for _, path := range paths {
		if path.Path == "new.go" {
			renamed = path
		}
	}
	if renamed.OriginalPath != "old.go" || renamed.IndexStatus != "R" {
		t.Fatalf("rename = %+v", renamed)
	}
	if _, err := ParseDirtyStatus([]byte("bad\x00")); err == nil {
		t.Fatal("malformed status passed")
	}
}
