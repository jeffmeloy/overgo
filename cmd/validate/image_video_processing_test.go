package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// checkMediaMemoOptimisation holds one report owner to the optimisation the
// retained acquisition accepted: the persisted discovery memo is loaded and
// handed to the capability catalog, so no model is re-hashed to render a
// report or export a sample. It reads the source's syntax, not its bytes: the
// acceptance once pinned the whole file to a declared edit transition, and six
// later landings that reshaped the report for other reasons left the
// optimisation in force while the pin read the file as changed beyond review.
func checkMediaMemoOptimisation(path string, source []byte) error {
	syntax, err := parser.ParseFile(token.NewFileSet(), path, source, parser.SkipObjectResolution)
	if err != nil {
		return err
	}
	selector := func(node ast.Expr) string {
		call, isCall := node.(*ast.CallExpr)
		if !isCall {
			return ""
		}
		named, isSelector := call.Fun.(*ast.SelectorExpr)
		if !isSelector {
			return ""
		}
		owner, isIdent := named.X.(*ast.Ident)
		if !isIdent {
			return ""
		}
		return owner.Name + "." + named.Sel.Name
	}
	catalogs, memoed := 0, 0
	ast.Inspect(syntax, func(node ast.Node) bool {
		call, isCall := node.(*ast.CallExpr)
		if !isCall || selector(call) != "discovery.CapabilityCatalogForTasks" {
			return true
		}
		catalogs++
		if len(call.Args) > mediaCatalogMemoArgument && selector(call.Args[mediaCatalogMemoArgument]) == "discovery.LoadMemo" {
			memoed++
		}
		return true
	})
	if catalogs == 0 {
		return fmt.Errorf("%s: no capability catalog is read; the accepted optimisation has no site", path)
	}
	if memoed != catalogs {
		return fmt.Errorf("%s: %d of %d capability catalog reads take the persisted memo; the accepted optimisation is not in force", path, memoed, catalogs)
	}
	return nil
}

// mediaCatalogMemoArgument is the position of the memo in a capability catalog
// read: context, store, limit, memo, tasks.
const mediaCatalogMemoArgument = 3

// TestMediaReportOwnersKeepTheMemo holds the report owners to the optimisation
// the retained processing acquisition accepted at its own source: every
// capability catalog read still takes the persisted memo, and a source that
// drops it is refused whatever else it keeps. It reads only the tree, so the
// landing runs it; the acquisition itself is the cycle's.
func TestMediaReportOwnersKeepTheMemo(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	for _, owner := range []string{"cmd/validate/media.go", "cmd/validate/samples.go"} {
		current, err := os.ReadFile(filepath.Join(root, owner))
		if err != nil {
			t.Fatal(err)
		}
		if err := checkMediaMemoOptimisation(owner, current); err != nil {
			t.Fatal(err)
		}
		withoutMemo := bytes.ReplaceAll(current, []byte("discovery.LoadMemo(ctx, store)"), []byte("nil"))
		if bytes.Equal(withoutMemo, current) {
			t.Fatalf("%s: memo-removal mutation did not change the source", owner)
		}
		if err := checkMediaMemoOptimisation(owner, withoutMemo); err == nil {
			t.Fatalf("%s: removal of the accepted memo optimization admitted", owner)
		}
	}
}

type mediaProcessingObservation struct {
	Variant   string            `json:"variant"`
	State     string            `json:"state"`
	Head      artifact.CommitID `json:"store_head"`
	Sequence  uint64            `json:"store_sequence"`
	Wall      uint64            `json:"wall_ns"`
	Allocated uint64            `json:"allocated_bytes"`
	Calls     uint64            `json:"allocation_calls"`
	Peak      uint64            `json:"peak_host_bytes"`
	Report    string            `json:"report_sha256"`
	Samples   map[string]string `json:"samples"`
}
