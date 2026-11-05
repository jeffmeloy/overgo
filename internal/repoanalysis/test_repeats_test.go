package repoanalysis

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/types"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/testcheck"
)

// testRepeatMost is the most bodies of one skeleton the module's tests may hold: two copies stay readable beside each
// other, and a third is a body to fold into one the cases share.
const testRepeatMost = 2

// testRepeatDeferred are the directories whose tests fold with consolidate/cell-step-reads, the next landing that
// runs the device lane in full, as any change to them does.
var testRepeatDeferred = []string{"internal/cuda/driver", "internal/cuda/executor"}

// boundTests are the test files a claim of compatibility.json names as its evidence or a record of docs/verification
// pins by digest: each changes with a verification of its own, so a fold leaves it.
func boundTests(t *testing.T, root string) map[string]bool {
	t.Helper()
	bound := map[string]bool{}
	data, err := os.ReadFile(filepath.Join(root, "compatibility.json"))
	testcheck.NoError(t, err)
	var manifest struct {
		Claims []struct {
			Evidence []struct {
				Path string `json:"path"`
			} `json:"evidence"`
		} `json:"claims"`
	}
	testcheck.NoError(t, json.Unmarshal(data, &manifest))
	for _, claim := range manifest.Claims {
		for _, evidence := range claim.Evidence {
			bound[evidence.Path] = true
		}
	}
	records, err := filepath.Glob(filepath.Join(root, "docs", "verification", "*.json"))
	testcheck.NoError(t, err)
	for _, record := range records {
		data, err := os.ReadFile(record)
		testcheck.NoError(t, err)
		var pinned struct {
			Inputs map[string]string `json:"Inputs"`
		}
		if json.Unmarshal(data, &pinned) == nil {
			for input := range pinned.Inputs {
				bound[filepath.ToSlash(input)] = true
			}
		}
	}
	return bound
}

// testBodies reads the function and closure bodies of repeatLeast nodes or more of the module's test files by skeleton,
// each package's tests type-checked as its test binary builds them: the in-package tests with the package's files, the
// external tests against that build.
func testBodies(t *testing.T, root string) map[string][]bodySite {
	t.Helper()
	load, err := loadModule(t.Context(), root)
	testcheck.NoError(t, err)
	bound := boundTests(t, root)
	var guard sync.Mutex
	var group sync.WaitGroup
	skeletons := map[string][]bodySite{}
	failures := []error{}
	for importPath, checked := range load.module {
		group.Go(func() {
			testFiles, err := load.parse(checked.listed, checked.listed.TestGoFiles)
			external, externalErr := load.parse(checked.listed, checked.listed.XTestGoFiles)
			if err != nil || externalErr != nil {
				guard.Lock()
				failures = append(failures, err, externalErr)
				guard.Unlock()
				return
			}
			read := func(files []*ast.File, info *types.Info) {
				for _, file := range files {
					relative, _ := filepath.Rel(load.root, load.fset.Position(file.Pos()).Filename)
					relative = filepath.ToSlash(relative)
					if bound[relative] || slices.Contains(testRepeatDeferred, filepath.ToSlash(filepath.Dir(relative))) {
						continue
					}
					ast.Inspect(file, func(node ast.Node) bool {
						var body *ast.BlockStmt
						switch typed := node.(type) {
						case *ast.FuncDecl:
							body = typed.Body
						case *ast.FuncLit:
							body = typed.Body
						}
						if body == nil {
							return true
						}
						if skeleton, values, nodes := bodyShape(body, info); nodes >= repeatLeast {
							place := load.fset.Position(body.Pos())
							relative, _ := filepath.Rel(load.root, place.Filename)
							guard.Lock()
							skeletons[skeleton] = append(skeletons[skeleton], bodySite{fmt.Sprintf("%s:%d", filepath.ToSlash(relative), place.Line), nodes, values})
							guard.Unlock()
						}
						return true
					})
				}
			}
			built := checked.pkg
			if len(testFiles) != 0 {
				var info *types.Info
				built, info, _ = load.check(checked.listed, importPath, append(slices.Clone(checked.files), testFiles...), nil)
				read(testFiles, info)
			}
			if len(external) != 0 {
				_, info, _ := load.check(checked.listed, importPath+"_test", external, built)
				read(external, info)
			}
		})
	}
	group.Wait()
	for _, failure := range failures {
		testcheck.NoError(t, failure)
	}
	return skeletons
}

// TestNoTestBodyRepeatsThreeTimes holds the module's tests to testRepeatMost bodies of one skeleton, read as the
// production repeat checks read theirs, so a body a third case repeats is folded into one the cases share and named
// for them. It reads the tests that change freely: not those bound by a claim or a record (boundTests), nor those of
// testRepeatDeferred.
func TestNoTestBodyRepeatsThreeTimes(t *testing.T) {
	t.Parallel()
	groups, _ := bodyRepeats(testBodies(t, checkoutRoot), 0)
	over := 0
	for _, group := range groups {
		if len(group)-1 > testRepeatMost {
			over++
			t.Errorf("%s nodes x%d: %s", group[0], len(group)-1, strings.Join(group[1:], " "))
		}
	}
	if over > 0 {
		t.Errorf("%d test bodies stand more than %d times, so fold each into one body its cases share", over, testRepeatMost)
	}
}
