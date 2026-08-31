package gate

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codeprofile"
	"overgo/internal/runrecord"
)

func TestImpactExclusionRateEvidence(t *testing.T) {
	checks := []automationcheck.Check{
		impactMetricCheck("first", "owner:first", "internal/first"),
		impactMetricCheck("second", "owner:second", "internal/second"),
	}
	impact := automationcheck.OwnershipImpact(checks, automationcheck.Surface{
		Identity: "snapshot", Packages: []string{"internal/second"},
	})
	metrics := automationcheck.MeasureSelection(checks, impact)
	selection := codeprofile.ImpactSelection{
		Identity: "snapshot", Owned: metrics.Owned, Triggered: metrics.Triggered,
		Excluded: metrics.Excluded, Unresolved: metrics.Unresolved,
	}
	if selection.Owned != 2 || selection.Triggered != 1 || selection.Excluded != 1 || selection.Unresolved != 0 ||
		!strings.Contains(impactSelectionHonesty(selection), "excluded=1/2") {
		t.Fatalf("impact selection = %+v, %q", selection, impactSelectionHonesty(selection))
	}
	gate, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("gate"))
	if err != nil {
		t.Fatal(err)
	}
	profile := codeProfileFixture()
	profile.Impact = selection
	evidence, err := codeprofile.NewEvidence("0123456789abcdef0123456789abcdef01234567", gate, profile)
	if err != nil {
		t.Fatal(err)
	}
	content, err := evidence.Content()
	if err != nil || !strings.Contains(string(content.Data), `"impact_selection":{"identity":"snapshot","owned":2,"triggered":1,"excluded":1,"unresolved":0}`) {
		t.Fatalf("published impact evidence = %s, %v", content.Data, err)
	}
}

func TestNoPathOnlySelection(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	allowed := []string{"MergeImpact", "OwnershipImpact"}
	entries, err := os.ReadDir(filepath.Join(root, "internal", "automationcheck"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		file, parseErr := parser.ParseFile(token.NewFileSet(), filepath.Join(root, "internal", "automationcheck", entry.Name()), nil, 0)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		for _, declaration := range file.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || !ast.IsExported(function.Name.Name) || !strings.HasSuffix(function.Name.Name, "Impact") || slices.Contains(allowed, function.Name.Name) {
				continue
			}
			t.Errorf("path-era impact producer %s remains in %s", function.Name.Name, entry.Name())
		}
	}
	mainFile, err := parser.ParseFile(token.NewFileSet(), filepath.Join(root, "cmd", "gate", "main.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, declaration := range mainFile.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Name.Name != "pipeline" {
			continue
		}
		ast.Inspect(function.Body, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			owner, packageCall := selector.X.(*ast.Ident)
			if packageCall && owner.Name == "automationcheck" && strings.HasSuffix(selector.Sel.Name, "Impact") && selector.Sel.Name != "OwnershipImpact" {
				t.Errorf("pipeline selects checks through %s", selector.Sel.Name)
			}
			return true
		})
	}
}

func impactMetricCheck(name string, fact automationcheck.Fact, packagePath string) automationcheck.Check {
	return automationcheck.Check{Descriptor: automationcheck.Descriptor{
		Name: name, Phase: runrecord.PhaseValidate, Triggers: []automationcheck.Fact{fact},
		Inapplicable: "disjoint ownership", Ownership: automationcheck.Ownership{Fact: fact, Packages: []string{packagePath}},
	}, Run: func(_ context.Context, _ automationcheck.Invocation) (bool, string, error) { return false, "", nil }}
}
