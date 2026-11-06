package overgodb

import (
	"bytes"
	"encoding/json"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// crashpointsTag is the build tag under which the crash recorder (crash_record_test.go) and the crash harness
// (crash_points_test.go) are compiled.
const crashpointsTag = "crashpoints"

// TestEveryDurableOperationIsACrashPoint records the store's durable operations through a workload of commits, seals,
// closes, a release and a pack, and reopens the store from every tree a crash at each operation can leave: a killed
// process, a power loss on a journaled volume, and the rule fsatomic writes for Windows. Every state holds the store's
// promises: no acknowledged commit lost, the journal alone replays to the open's head, and the store goes on. The store's
// production files are rewritten in an overlay so that every durable call reaches the recorder, which holds nothing of
// production code; a durable call the rewrite misses fails the harness, since the recorded operations then do not
// rebuild the tree the workload left. It also holds an append to streaming its frame as three writes and a flush.
func TestEveryDurableOperationIsACrashPoint(t *testing.T) {
	// Serial: it runs the package's tests again in a go test of its own, which compiles the package under an overlay.
	runCrashHarness(t, "TestCrashStatesKeepTheStoresPromises", "TestLogAppendStreamsFrame")
}

// TestEveryDurableOperationCanFail refuses each durable operation of the workload in turn, once and from there on, as a
// full volume would, lets the workload go on as a process would, and holds the store it leaves to the same promises.
// A part-way append leaves the head and catalog as they were and its blob uncommitted, every surface refusing until a
// reopen, and a refused flush replays the whole uncertain commit.
func TestEveryDurableOperationCanFail(t *testing.T) {
	// Serial: it runs the package's tests again in a go test of its own, which compiles the package under an overlay.
	runCrashHarness(t, "TestRefusedOperationsKeepTheStoresPromises", "TestAppendFaultClosesEveryStoreSurfaceUntilReopen",
		"TestSyncFaultReplaysCompleteUncertainCommit")
}

// TestGeneratedStoreLifecycles drives seeded lifecycles of commits, seals, closes, opens, reads, releases, packs, repeats
// and stale heads across two writers and a reader, holds every handle to a small reference of keys, contents and head,
// and holds a sample of the states a kill leaves in each to the store's promises.
func TestGeneratedStoreLifecycles(t *testing.T) {
	// Serial: it runs the package's tests again in a go test of its own, which compiles the package under an overlay.
	runCrashHarness(t, "TestLifecyclesKeepTheStoresPromises")
}

// runCrashHarness runs tests of the crash harness in a go test of their own, over the overlay crashOverlay writes, and
// fails unless each of them ran and passed.
func runCrashHarness(t *testing.T, harnesses ...string) {
	t.Helper()
	overlay := crashOverlay(t)
	command := exec.CommandContext(t.Context(), "go", "test", "-count=1", "-tags", crashpointsTag, "-overlay", overlay,
		"-run", "^("+strings.Join(harnesses, "|")+")$", "-v", ".")
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("the crash harness fails: %v\n%s", err, output)
	}
	passed := map[string]bool{}
	for line := range strings.Lines(string(output)) {
		if name, found := strings.CutPrefix(strings.TrimSpace(line), "--- PASS: "); found {
			passed[strings.Fields(name)[0]] = true
		}
		if strings.Contains(line, "crash_points_test.go") || strings.Contains(line, "crash_lifecycles_test.go") {
			t.Log(strings.TrimSpace(line))
		}
	}
	for _, harness := range harnesses {
		if !passed[harness] {
			t.Errorf("the crash harness ran no test %s", harness)
		}
	}
}

// crashOverlay writes, under a directory of the test's, each production file of the package whose durable calls it
// rewrites, and the go test overlay that puts them in place of the originals, and returns the overlay's path.
func crashOverlay(t *testing.T) string {
	t.Helper()
	directory := testcheck.Must(os.Getwd())(t)
	entries := testcheck.Must(os.ReadDir(directory))(t)
	out := t.TempDir()
	replace := map[string]string{}
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		files := token.NewFileSet()
		file := testcheck.Must(parser.ParseFile(files, filepath.Join(directory, name), nil, parser.ParseComments))(t)
		if !rewriteDurableCalls(file) {
			continue
		}
		dropUnusedImports(file)
		var text bytes.Buffer
		testcheck.NoError(t, format.Node(&text, files, file))
		rewritten := filepath.Join(out, name)
		testcheck.NoError(t, os.WriteFile(rewritten, text.Bytes(), storeFileMode))
		replace[filepath.Join(directory, name)] = rewritten
	}
	if len(replace) == 0 {
		t.Fatal("no production file of the package makes a durable call")
	}
	document := testcheck.Must(json.Marshal(map[string]any{"Replace": replace}))(t)
	overlay := filepath.Join(out, "overlay.json")
	testcheck.NoError(t, os.WriteFile(overlay, document, storeFileMode))
	return overlay
}

// The calls the rewrite sends through the recorder: the os and fsatomic functions that change the namespace, each by
// the name the recorder gives it, and the methods that write or flush a file, by the names the store's files give the
// values they call them on.
var (
	crashPackageCalls = map[string]map[string]string{
		"os": {
			"RemoveAll": "RemoveAll", "Rename": "Rename", "Remove": "Remove", "MkdirAll": "MkdirAll", "Mkdir": "Mkdir",
			"CreateTemp": "CreateTemp", "OpenFile": "OpenFile", "Link": "Link",
		},
		"fsatomic": {"Replace": "Replace", "Remove": "FSRemove", "SyncDirectory": "SyncDirectory", "SyncFile": "SyncFile"},
	}
	crashSyncReceivers     = []string{"file", "staging", "output", "writer"}
	crashTruncateReceivers = []string{"file"}
	crashWriteReceivers    = []string{"writer"}
)

// rewriteDurableCalls sends file's durable calls through the crash recorder, in place, and says whether it changed any.
func rewriteDurableCalls(file *ast.File) bool {
	changed := false
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		method := selector.Sel.Name
		if owner, ok := selector.X.(*ast.Ident); ok {
			if renamed, durable := crashPackageCalls[owner.Name][method]; durable {
				selector.X, selector.Sel = ast.NewIdent("crashos"), ast.NewIdent(renamed)
				changed = true
				return true
			}
			if owner.Name == "io" && method == "Copy" {
				call.Fun, changed = ast.NewIdent("crashCopy"), true
				return true
			}
			if owner.Name == "io" && method == "MultiWriter" {
				for index, argument := range call.Args {
					call.Args[index] = &ast.CompositeLit{Type: ast.NewIdent("crashTee"), Elts: []ast.Expr{argument}}
				}
				changed = true
				return true
			}
		}
		receiver := lastName(selector.X)
		switch {
		case method == "Sync" && len(call.Args) == 0 && slices.Contains(crashSyncReceivers, receiver):
			call.Fun, call.Args = ast.NewIdent("crashSync"), []ast.Expr{selector.X}
		case method == "Truncate" && len(call.Args) == 1 && slices.Contains(crashTruncateReceivers, receiver):
			call.Fun, call.Args = ast.NewIdent("crashTruncate"), append([]ast.Expr{selector.X}, call.Args...)
		case method == "Write" && len(call.Args) == 1 && slices.Contains(crashWriteReceivers, receiver):
			call.Fun, call.Args = ast.NewIdent("crashWrite"), append([]ast.Expr{selector.X}, call.Args...)
		default:
			return true
		}
		changed = true
		return true
	})
	return changed
}

// lastName is the name an expression's chain of selections ends in: file for s.log.file, and nothing for a call.
func lastName(expression ast.Expr) string {
	switch typed := expression.(type) {
	case *ast.Ident:
		return typed.Name
	case *ast.SelectorExpr:
		return typed.Sel.Name
	}
	return ""
}

// dropUnusedImports removes the imports the rewrite left file no selection of.
func dropUnusedImports(file *ast.File) {
	used := map[string]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok {
			if owner, ok := selector.X.(*ast.Ident); ok {
				used[owner.Name] = true
			}
		}
		return true
	})
	for _, declaration := range file.Decls {
		group, ok := declaration.(*ast.GenDecl)
		if !ok || group.Tok != token.IMPORT {
			continue
		}
		group.Specs = slices.DeleteFunc(group.Specs, func(spec ast.Spec) bool {
			imported := spec.(*ast.ImportSpec)
			path, err := strconv.Unquote(imported.Path.Value)
			if err != nil {
				return false
			}
			name := path[strings.LastIndex(path, "/")+1:]
			if imported.Name != nil {
				name = imported.Name.Name
			}
			return name != "_" && !used[name]
		})
	}
	file.Imports = slices.DeleteFunc(file.Imports, func(imported *ast.ImportSpec) bool {
		return !slices.ContainsFunc(file.Decls, func(declaration ast.Decl) bool {
			group, ok := declaration.(*ast.GenDecl)
			return ok && slices.Contains(group.Specs, ast.Spec(imported))
		})
	})
}
