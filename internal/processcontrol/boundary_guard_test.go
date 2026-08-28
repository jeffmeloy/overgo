package processcontrol

import (
	"go/ast"
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

// execExceptions grandfathers the production files still holding
// direct process execution, each with the reason and the work that
// retires it. The list only shrinks: a new direct exec or kill site
// anywhere else refuses, and a stale entry whose sites are gone
// refuses too, so the boundary ratchets toward one owner.
var execExceptions = map[string]string{
	"internal/runrecord/verifying_commit.go":       "read-only git fact reads; execution-semantics binds repository facts to supervised receipts",
	"internal/repoanalysis/build.go":               "read-only go toolchain fact reads; execution-semantics migration",
	"internal/repoanalysis/imports.go":             "read-only go toolchain fact reads; execution-semantics migration",
	"internal/clioptions/command.go":               "command self-identification fact; execution-semantics migration",
	"internal/testutil/process.go":                 "test-support process helpers consumed only by suites",
	"cmd/plan/main.go":                             "plan verify and dispatch tooling; execution-semantics migration",
	"cmd/plan/sync.go":                             "read-only git fact reads; execution-semantics migration",
	"cmd/gate/main.go":                             "gate check invocations; execution-semantics binds gate steps to supervised receipts",
	"cmd/loophook/main.go":                         "hook-time tooling; execution-semantics migration",
	"cmd/release/main.go":                          "release lane tooling; execution-semantics migration",
	"cmd/reverify/main.go":                         "reverify lane tooling; execution-semantics migration",
	"cmd/build-kernels/main.go":                    "kernel toolchain invocations; execution-semantics migration",
	"cmd/smoke-lane/main.go":                       "smoke lane tooling; execution-semantics migration",
	"cmd/test-lane/main.go":                        "test lane tooling; execution-semantics migration",
	"cmd/eval-lane/main.go":                        "eval lane tooling; execution-semantics migration",
	"cmd/evaluate/main.go":                         "evaluation tooling; execution-semantics migration",
	"cmd/evaluate/all.go":                          "evaluation tooling; execution-semantics migration",
	"cmd/sbom/main.go":                             "read-only go toolchain fact reads; execution-semantics migration",
	"cmd/recipe/main.go":                           "read-only git fact reads; execution-semantics migration",
	"cmd/advisories/main.go":                       "read-only git fact reads; execution-semantics migration",
	"cmd/compatibility/training.go":                "training lane tooling; execution-semantics migration",
	"cmd/composite-generation-lane/run_windows.go": "generation lane tooling; execution-semantics migration",
}

// TestProductionCommandsUseSupervisor is the process boundary: outside
// this package, production code neither spawns processes through
// os/exec nor kills them directly. Grandfathered files are enumerated
// with retirement reasons and must still hold a site, so the exception
// list can only shrink.
func TestProductionCommandsUseSupervisor(t *testing.T) {
	root := filepath.Join("..", "..")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]int{}
	for _, file := range snapshot.Files {
		if file.Test || filepath.ToSlash(filepath.Dir(file.Path)) == "internal/processcontrol" {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			t.Fatal(err)
		}
		sites := 0
		ast.Inspect(syntax, func(node ast.Node) bool {
			selector, ok := node.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if base, ok := selector.X.(*ast.Ident); ok && base.Name == "exec" &&
				(selector.Sel.Name == "Command" || selector.Sel.Name == "CommandContext") {
				sites++
			}
			if selector.Sel.Name == "Kill" {
				if inner, ok := selector.X.(*ast.SelectorExpr); ok && inner.Sel.Name == "Process" {
					sites++
				}
			}
			return true
		})
		if sites == 0 {
			continue
		}
		path := filepath.ToSlash(file.Path)
		if _, grandfathered := execExceptions[path]; !grandfathered {
			t.Errorf("%s spawns or kills processes directly; route external execution through processcontrol", path)
			continue
		}
		found[path] += sites
	}
	for path := range execExceptions {
		if found[path] == 0 {
			t.Errorf("exception %s no longer holds a direct exec site; delete its entry", path)
		}
	}
}
