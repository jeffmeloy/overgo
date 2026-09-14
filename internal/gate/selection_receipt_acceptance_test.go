package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/automationcheck"
)

// TestSelectionReceiptAcceptance compares lane admission and receipt validity
// against a real failing file consumer and an independent temporary-file test.
func TestSelectionReceiptAcceptance(t *testing.T) {
	t.Parallel()
	g := scopeCompilerFixture(t)
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/named/named_test.go", `package named
import ("os"; "testing")
func TestRead(t *testing.T) {
 data, err := os.ReadFile("../../internal/other/other.go")
 if err != nil || string(data) != "package other\n" { t.Fatalf("changed input: %s %v", data, err) }
}
`)
	write("internal/temporary/temporary_test.go", `package temporary
import ("os"; "path/filepath"; "testing")
func TestWrite(t *testing.T) {
 if err := os.WriteFile(filepath.Join(t.TempDir(), "output"), []byte("ok"), 0600); err != nil { t.Fatal(err) }
}
`)
	runGitFixture(t, g.repo, "add", ".")
	g.packageGraph = nil
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	packages := []string{"overgo/internal/named", "overgo/internal/temporary"}
	before, err := packageInputIdentities(graph, packages)
	if err != nil {
		t.Fatal(err)
	}
	if output, err := command(g.repo, "go", "test", "./internal/named", "./internal/temporary", "-count=1"); err != nil {
		t.Fatalf("baseline: %s %v", output, err)
	}
	cache := automationcheck.NewEvidenceCache(lifecycleTestEnvironment(t).ID)
	for _, name := range packages {
		if err := cache.RecordPackagePass(name, "complete", before[name]); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/other/other.go", "package other\n// changed input\n")
	g.paths = []string{"internal/other/other.go"}
	g.packageGraph = nil
	resolver, err := g.dependencyResolver()
	if err != nil {
		t.Fatal(err)
	}
	if !resolver(automationcheck.Ownership{Packages: []string{"internal/missing"}}, "internal/other") {
		t.Fatal("unknown lane owner was excluded")
	}
	after, err := packageInputIdentities(*g.packageGraph, packages)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range packages {
		wantRun := name == "overgo/internal/named"
		owned := automationcheck.Ownership{Packages: []string{strings.TrimPrefix(name, "overgo/")}}
		if got := resolver(owned, "internal/other"); got != wantRun {
			t.Errorf("lane %s selected=%t want %t", name, got, wantRun)
		}
		if reused, err := cache.PackageReusable(name, "complete", after[name]); err != nil || reused == wantRun {
			t.Errorf("receipt %s reused=%t err=%v", name, reused, err)
		}
	}
	for _, targets := range [][]string{{"./internal/named", "./internal/temporary"}, {"./internal/named"}} {
		args := append([]string{"test", "-count=1"}, targets...)
		output, err := command(g.repo, "go", args...)
		if err == nil || !strings.Contains(output, "changed input") {
			t.Fatalf("scope %v missed seeded failure: %s %v", targets, output, err)
		}
	}
	// Deleted inputs retain broad selection until their former owner resolves.
	if err := os.Remove(filepath.Join(g.repo, "internal/other/other.go")); err != nil {
		t.Fatal(err)
	}
	resolver, err = g.dependencyResolver()
	if err != nil || !resolver(automationcheck.Ownership{Packages: []string{"internal/temporary"}}, "internal/other") {
		t.Fatalf("removed input excluded a lane: %v", err)
	}
	t.Log("full=2 selected=1 exact receipts reused=1; both executions detect the same seeded failure; removed input remains conservative; fixture comparison, no wall-speedup claim")
}
