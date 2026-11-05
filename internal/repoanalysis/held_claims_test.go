package repoanalysis

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// anchoredDeclaration names the function or method a claim's evidence anchors, its contains being the declaration's
// opening up to its parameters ("func (r *Runner) Encode("), as the census names it ("Runner.Encode"); an anchor of
// anything else names none.
func anchoredDeclaration(contains string) (string, bool) {
	if !strings.HasPrefix(contains, "func ") || !strings.HasSuffix(contains, "(") {
		return "", false
	}
	file, err := parser.ParseFile(token.NewFileSet(), "", "package anchor\n"+contains+") {}", parser.SkipObjectResolution)
	if err != nil || len(file.Decls) != 1 {
		return "", false
	}
	function, isFunction := file.Decls[0].(*ast.FuncDecl)
	if !isFunction {
		return "", false
	}
	return declarationName(function), true
}

// TestNoTestHeldEntryWaitsOnAClaim holds that no claim of compatibility.json rests its source on a declaration only
// tests reach: a source evidence whose contains opens a function or method must not anchor one the census keeps as
// test-held, since a claim proves code a command runs, and an entry it anchored could not go with its tests.
func TestNoTestHeldEntryWaitsOnAClaim(t *testing.T) {
	t.Parallel()
	for contains, want := range map[string]string{
		"func (r *Runner) Encode(":      "Runner.Encode",
		"func (s Store[T]) Load(":       "Store.Load",
		"func Compile(":                 "Compile",
		"type Runner struct":            "",
		"func TestEncodeMatchesGolden(": "TestEncodeMatchesGolden",
	} {
		if got, ok := anchoredDeclaration(contains); got != want || ok != (want != "") {
			t.Fatalf("the anchor %q names %q (%v), want %q", contains, got, ok, want)
		}
	}
	_, classOf, err := sortedEntries(keptList(t, unreachedList))
	testcheck.NoError(t, err)
	data, err := os.ReadFile(filepath.Join(checkoutRoot, "compatibility.json"))
	testcheck.NoError(t, err)
	var manifest struct {
		Claims []struct {
			ID       string `json:"id"`
			Evidence []struct {
				Contains string `json:"contains"`
				Path     string `json:"path"`
				Role     string `json:"role"`
			} `json:"evidence"`
		} `json:"claims"`
	}
	testcheck.NoError(t, json.Unmarshal(data, &manifest))
	anchors := 0
	for _, claim := range manifest.Claims {
		for _, evidence := range claim.Evidence {
			name, ok := anchoredDeclaration(evidence.Contains)
			if evidence.Role != "source" || !ok {
				continue
			}
			anchors++
			if entry := evidence.Path + " " + name; classOf[entry] == "test-held" {
				t.Errorf("the claim %s anchors %s, which %s keeps as test-held, so anchor the live code the claim proves, or give the entry a command", claim.ID, entry, unreachedList)
			}
		}
	}
	if anchors == 0 {
		t.Fatal("compatibility.json anchors no claim on a declaration, so the check reads nothing")
	}
}
