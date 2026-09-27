package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/gosource"
	"overgo/internal/gosyntax"
)

// TestSelectionStaysNarrowForGateAndRemovedCode holds two changes to the
// checks they can reach. A census change is an ordinary change: the manifest
// owners do not import the repository censuses, so it leaves the impact
// analysis trusted, while a change to the source snapshot they parse does
// not. A removed Go file reaches what its package reaches: a lane compiling
// the package is selected, and one compiling neither it nor an importer is
// not.
func TestSelectionStaysNarrowForGateAndRemovedCode(t *testing.T) {
	t.Parallel()
	if requiresManifestBootstrap([]string{"internal/repoanalysis/tracked_documents.go"}) {
		t.Fatal("a census change forced the complete plan")
	}
	if !requiresManifestBootstrap([]string{"internal/gosyntax/snapshot.go"}) {
		t.Fatal("a change to the parsed source snapshot left the impact analysis trusted")
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	deps, err := command(root, "go", "list", "-deps", "overgo/cmd/code-manifest", "overgo/internal/automationcheck")
	if err != nil {
		t.Fatal(err)
	}
	for dep := range strings.FieldsSeq(deps) {
		if strings.HasPrefix(dep, "overgo/internal/repoanalysis") {
			t.Fatalf("the manifest owners import %s, so a census change must bootstrap the manifest", dep)
		}
	}

	g := runtimeReaderFixture(t)
	const removed = "internal/unrelated/removed.go"
	if _, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(removed))); !os.IsNotExist(err) {
		t.Fatalf("the removed file exists: %v", err)
	}
	g.paths = []string{removed}
	resolver, err := g.dependencyResolver()
	if err != nil {
		t.Fatal(err)
	}
	owning := func(packages ...string) automationcheck.Ownership {
		return automationcheck.Ownership{Fact: "lane", Packages: packages}
	}
	if !resolver(owning("internal/isolated"), "internal/unrelated") {
		t.Fatal("a lane compiling the removed file's package was not selected")
	}
	if resolver(owning("internal/readerclient"), "internal/unrelated") {
		t.Fatal("a removed Go file selected a lane that compiles neither its package nor an importer")
	}

	// The base manifest reads the candidate tree's selection, which has no
	// decision for a removed file: it is decided by its own constraints in
	// its siblings' package, and one whose package is gone stays undecided.
	selection, err := gosource.HostBuildSelection(g.repo, "./internal/...")
	if err != nil {
		t.Fatal(err)
	}
	const excluded, gone = "internal/unrelated/removed_plan9.go", "internal/gone/gone.go"
	base, err := g.source.Overlay(map[string][]byte{
		removed:  []byte("package unrelated\n\nfunc Removed() int { return Value }\n"),
		excluded: []byte("package unrelated\n"),
		gone:     []byte("package gone\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	covered, err := gosyntax.Cover(selection, base)
	if err != nil {
		t.Fatal(err)
	}
	if !covered.Files[removed] || covered.Packages[removed] != "overgo/internal/unrelated" {
		t.Fatalf("removed file decision = %v in %q", covered.Files[removed], covered.Packages[removed])
	}
	if selected, known := covered.Files[excluded]; !known || selected {
		t.Fatalf("removed file for another GOOS decided=%v selected=%v", known, selected)
	}
	if _, known := covered.Files[gone]; known {
		t.Fatal("a file whose package is gone was decided")
	}
	if _, known := selection.Files[removed]; known {
		t.Fatal("covering the base changed the candidate selection")
	}
}
