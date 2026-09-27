package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/automationcheck"
)

// TestDeletedDocumentSelectsItsReaders holds a deleted document to the lanes
// that reach it: a package that still names it is its reader in the
// candidate, so a lane compiling that reader is selected and a lane reaching
// neither the name nor the reader is not, as for a modified document; the
// deletion alone no longer selects every lane.
func TestDeletedDocumentSelectsItsReaders(t *testing.T) {
	t.Parallel()
	g := runtimeReaderFixture(t)
	const document = "docs/config.txt"
	if err := os.Remove(filepath.Join(g.repo, filepath.FromSlash(document))); err != nil {
		t.Fatal(err)
	}
	g.paths = []string{document}
	g.packageGraph = nil
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	readers := graph.namedReaders(document)
	if !slices.Equal(readers, []string{"internal/reader"}) {
		t.Fatalf("named readers of the deleted document = %v", readers)
	}
	g.attributedDocuments = map[string][]string{document: readers}
	resolver, err := g.dependencyResolver()
	if err != nil {
		t.Fatal(err)
	}
	owning := func(packages ...string) automationcheck.Ownership {
		return automationcheck.Ownership{Fact: "lane", Packages: packages}
	}
	if !resolver(owning("internal/readerclient"), "docs") {
		t.Fatal("a lane compiling the deleted document's reader was not selected")
	}
	if resolver(owning("internal/isolated"), "docs") {
		t.Fatal("deleting a document selected a lane that reaches neither its name nor its reader")
	}
}
