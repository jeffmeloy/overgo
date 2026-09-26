package repoanalysis

import (
	"os"
	"path"
	"path/filepath"
	"testing"
)

// TestRootHoldsNoPolicyDocuments holds the repository root to documents it
// shares: a JSON document there that one package alone reads and no tool
// writes is that package's own input, and it lives beside the package that
// embeds it.
func TestRootHoldsNoPolicyDocuments(t *testing.T) {
	t.Parallel()
	root := "../.."
	snapshot, err := DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	documents, err := TrackedDocuments(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	census, err := TrackedDocumentCensus(snapshot, documents)
	if err != nil {
		t.Fatal(err)
	}
	shared := 0
	for _, document := range census {
		if path.Dir(document.Path) != "." || path.Ext(document.Path) != ".json" {
			continue
		}
		if len(document.Readers) == 1 && len(document.Writers) == 0 {
			t.Errorf("%s is read by %s alone; embed it beside that package", document.Path, document.Readers[0])
			continue
		}
		shared++
	}
	if shared == 0 {
		if _, err := os.Stat(filepath.Join(root, "compatibility.json")); err == nil {
			t.Fatal("no shared root document was measured")
		}
	}
}
