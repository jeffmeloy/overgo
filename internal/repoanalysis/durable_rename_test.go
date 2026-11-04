package repoanalysis

import (
	"go/ast"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// renameOwners are the production files outside internal/fsatomic that rename a path with os.Rename, each with why the
// name it gives need not survive a crash. A durable name, one whose loss loses history, is published through
// fsatomic.Replace or fsatomic.PublishDirectory, which make it reach the disk: on Windows fsatomic.SyncDirectory does
// nothing, so a bare rename's name can be lost while what follows it is kept.
var renameOwners = map[string]string{
	"cmd/loophook/main.go": "moves a session's stop block aside for its next prompt to report, a notice a crash only loses",
	"internal/controlleraction/candidate_materialization.go": "moves a merged candidate from its private stage to its output, which a crash leaves to be merged again",
	"internal/hfhub/download.go":                             "names a download its digest has checked; a crash leaves the partial file, which the next download replaces",
	"internal/overgodb/checkpoints.go":                       "names a projection checkpoint, derived state that open discards for a replay when it is missing",
	"internal/overgodb/store.go":                             "the cutover's renames, which store-durability/cutover-kill keeps from starting an empty store",
	"internal/processlock/state.go":                          "carries a process-state file to its directory and names the directory's ignore rule, neither of them history",
	"internal/safetensors/stream_writer.go":                  "publishes a conversion's files, which a crash leaves to be converted again",
	"internal/safetensors/write.go":                          "names a written safetensors file, which a crash leaves to be written again",
}

// cutoverRenames are ReplaceStore's renames: the live store aside, the candidate in, and the live store back on a failure.
const cutoverRenames = 3

// TestEveryDurableRenameGoesThroughItsOwner holds every production os.Rename outside internal/fsatomic to a declared
// owner whose name need not survive a crash: a store's sealed segment and a training checkpoint publish through
// fsatomic, and a rename added beside them is refused until it does or says why it need not. A declaration whose file no
// longer renames is stale. A rename planted in a file of its own is found.
func TestEveryDurableRenameGoesThroughItsOwner(t *testing.T) {
	t.Parallel()
	sites := renameSites(t, "../..")
	for _, path := range slices.Sorted(maps.Keys(sites)) {
		if renameOwners[path] == "" {
			t.Errorf("%s renames with os.Rename %d times; publish a durable name through fsatomic.Replace or declare why it need not survive a crash", path, sites[path])
		}
	}
	for _, path := range slices.Sorted(maps.Keys(renameOwners)) {
		if sites[path] == 0 {
			t.Errorf("%s is declared a rename owner but no longer renames with os.Rename; drop the declaration", path)
		}
	}
	// The store's renames are the cutover's three; its sealed segments publish through fsatomic.Replace.
	if sites["internal/overgodb/store.go"] != cutoverRenames {
		t.Errorf("internal/overgodb/store.go renames with os.Rename %d times, want the cutover's %d", sites["internal/overgodb/store.go"], cutoverRenames)
	}
	planted := t.TempDir()
	file := filepath.Join(planted, "internal", "planted", "planted.go")
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package planted\n\nimport moved \"os\"\n\nfunc publish() error { return moved.Rename(\"a\", \"b\") }\n"
	if err := os.WriteFile(file, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	if found := renameSites(t, planted); found["internal/planted/planted.go"] != 1 {
		t.Fatalf("a rename planted under another name for os was found %d times, want once", found["internal/planted/planted.go"])
	}
}

// renameSites counts, by file, the os.Rename calls in the production Go files of internal and cmd under root, outside
// internal/fsatomic, whatever name the file gives the os package.
func renameSites(t *testing.T, root string) map[string]int {
	t.Helper()
	snapshot, err := gosyntax.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	sites := map[string]int{}
	for _, file := range snapshot.Files {
		if file.Test || strings.HasPrefix(file.Path, "internal/fsatomic/") {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			t.Fatal(err)
		}
		local := ""
		for _, spec := range syntax.Imports {
			if path, _ := strconv.Unquote(spec.Path.Value); path == "os" {
				local = "os"
				if spec.Name != nil {
					local = spec.Name.Name
				}
			}
		}
		if local == "" {
			continue
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			if selector, ok := node.(*ast.SelectorExpr); ok && selector.Sel.Name == "Rename" {
				if name, ok := selector.X.(*ast.Ident); ok && name.Name == local {
					sites[file.Path]++
				}
			}
			return true
		})
	}
	return sites
}
