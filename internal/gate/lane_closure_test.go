package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/automationcheck"
)

// TestLaneClosureAcceptance pins a lane's reach to what it compiles: a lane
// whose owned package imports a Go-parsing helper only in its tests leaves
// the scope of an unrelated source change, a lane whose production code
// compiles such a helper stays, and on the live graph a gate-only change
// excludes the device lane by closure while an automation-check change
// retains it. The browser lane's decision is measured and recorded.
func TestLaneClosureAcceptance(t *testing.T) {
	t.Run("fixture", laneClosureFixture)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	g := &gateContext{repo: root, paths: []string{"internal/gate/preflight.go"}}
	resolver, err := g.dependencyResolver()
	if err != nil {
		t.Fatal(err)
	}
	var checks []automationcheck.Check
	for _, check := range g.pipelineChecks() {
		if check.Descriptor.Ownership.Fact != "" {
			checks = append(checks, check)
		}
	}
	impact := automationcheck.OwnershipByDependency(checks, []string{"internal/gate"}, resolver)
	reason, excluded := impact.ExclusionReason("device")
	if !excluded || !strings.Contains(reason, "closure") {
		t.Fatalf("gate-only change reached the device lane: excluded=%v reason=%s", excluded, reason)
	}
	for _, name := range []string{automationcheck.WebUICheckName, "manifest", "published", "sbom", "claims"} {
		reason, excluded := impact.ExclusionReason(name)
		t.Logf("gate-only change: %s excluded=%v reason=%s", name, excluded, reason)
	}
	// A change the device packages compile keeps the lane; the analyzer
	// and check-definition owners keep every lane through the bootstrap
	// rule before this resolver is consulted.
	relevant := automationcheck.OwnershipByDependency(checks, []string{"internal/cuda/driver"}, resolver)
	if _, excluded := relevant.ExclusionReason("device"); excluded {
		t.Fatal("device driver change excluded the device lane")
	}
}

// laneClosureFixture: parser parses Go from an unnamed path, tested imports
// it only in tests, compiled imports it in production; a source change in
// other excludes the lane owning tested and keeps the lane owning compiled.
func laneClosureFixture(t *testing.T) {
	g := scopeCompilerFixture(t)
	write := func(name, content string) {
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/parser/parser.go", "package parser\nimport (\"go/parser\"; \"go/token\"; \"os\")\nfunc Count(path string) int { data, err := os.ReadFile(path); if err != nil { return -1 }; file, err := parser.ParseFile(token.NewFileSet(), path, data, 0); if err != nil { return -1 }; return len(file.Decls) }\n")
	write("internal/tested/tested.go", "package tested\nconst Value = 1\n")
	write("internal/tested/tested_test.go", "package tested\nimport (\"testing\"; \"overgo/internal/parser\")\nfunc TestValue(t *testing.T) { if parser.Count(\"tested.go\") < 0 { t.Skip() } }\n")
	write("internal/compiled/compiled.go", "package compiled\nimport \"overgo/internal/parser\"\nfunc Value() int { return parser.Count(\"compiled.go\") }\n")
	write("internal/compiled/compiled_test.go", "package compiled\nimport \"testing\"\nfunc TestValue(t *testing.T) { Value() }\n")
	runGitFixture(t, g.repo, "add", ".")
	g.packageGraph = nil
	resolver, err := g.dependencyResolver()
	if err != nil {
		t.Fatal(err)
	}
	owning := func(name string, fact automationcheck.Fact, packages ...string) automationcheck.Check {
		return automationcheck.Check{Descriptor: automationcheck.Descriptor{Name: name, Ownership: automationcheck.Ownership{Fact: fact, Packages: packages}}}
	}
	checks := []automationcheck.Check{
		owning("tested-lane", "tested fact", "internal/tested"),
		owning("compiled-lane", "compiled fact", "internal/compiled"),
	}
	impact := automationcheck.OwnershipByDependency(checks, []string{"internal/other"}, resolver)
	var excluded []string
	for _, exclusion := range impact.Exclusions {
		excluded = append(excluded, exclusion.Check)
	}
	if !slices.Equal(excluded, []string{"tested-lane"}) {
		t.Fatalf("source change outside both lanes: excluded=%v facts=%v, want only tested-lane", excluded, impact.Facts)
	}
	impact = automationcheck.OwnershipByDependency(checks, []string{"internal/parser"}, resolver)
	if len(impact.Exclusions) != 0 {
		t.Fatalf("helper change excluded a lane that compiles or tests with it: %+v", impact.Exclusions)
	}
}
