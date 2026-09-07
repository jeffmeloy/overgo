package gate

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/automationcheck"
)

func TestChangeSelectionExternalConsumers(t *testing.T) {
	t.Run("identity batch reuse stays fresh", func(t *testing.T) {
		g := scopeCompilerFixture(t)
		graph, err := g.inputGraph()
		if err != nil {
			t.Fatal(err)
		}
		packages := fixtureRootPackages(t, g)
		before, err := packageInputIdentities(graph, packages)
		if err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		for _, pkg := range packages {
			identity, err := graph.identity(pkg)
			if err != nil || identity != before[pkg] {
				t.Fatalf("standalone identity changed %s: %v", pkg, err)
			}
		}
		standaloneWall := time.Since(started)
		started = time.Now()
		batched, err := packageInputIdentities(graph, packages)
		if err != nil || !maps.Equal(before, batched) {
			t.Fatalf("batched identities changed: %v", err)
		}
		batchedWall := time.Since(started)
		cached := graph
		cached.fileInputs = map[string][]byte{}
		for _, pkg := range packages {
			identity, err := cached.identity(pkg)
			if err != nil || identity != before[pkg] {
				t.Fatalf("memoized identity changed %s: %v", pkg, err)
			}
		}
		unique := len(cached.fileInputs)
		catalog := filepath.Join(g.repo, "internal/plan/catalog.txt")
		if err := os.Remove(catalog); err != nil {
			t.Fatal(err)
		}
		for _, pkg := range packages {
			identity, err := cached.identity(pkg)
			if err != nil || identity != before[pkg] {
				t.Fatalf("repeated identity changed %s: %v", pkg, err)
			}
		}
		if unique == 0 || len(cached.fileInputs) != unique {
			t.Fatal("memoized file set changed")
		}
		after, err := packageInputIdentities(cached, packages)
		if err != nil || maps.Equal(before, after) || before["overgo/internal/client"] == after["overgo/internal/client"] {
			t.Fatalf("later batch retained stale input digest: %v", err)
		}
		fresh, err := graph.identity("overgo/internal/client")
		if err != nil || fresh != after["overgo/internal/client"] {
			t.Fatalf("standalone and batch identities differ: %v", err)
		}
		if err := os.WriteFile(catalog, []byte("changed between batches"), 0o644); err != nil {
			t.Fatal(err)
		}
		restored, err := packageInputIdentities(cached, packages)
		if err != nil || maps.Equal(after, restored) || maps.Equal(before, restored) {
			t.Fatalf("restored file retained stale identity: %v", err)
		}
		t.Logf("packages=%d distinct_cached_files=%d standalone_wall=%s batched_wall=%s; identical IDs, within-batch memo retained, later batches observe deletion and changed bytes", len(packages), unique, standaloneWall, batchedWall)
	})
	for _, change := range []string{"command output", "shared launcher", "renamed command", "deleted command", "unowned protocol file"} {
		t.Run(change, func(t *testing.T) {
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
			command := "package main\nimport \"fmt\"\nfunc main() { fmt.Print(1) }\n"
			if change == "unowned protocol file" {
				command = "package main\nimport (\"fmt\"; \"os\")\nfunc main() { b, err := os.ReadFile(\"../../protocol.txt\"); if err != nil { panic(err) }; fmt.Print(string(b)) }\n"
				write("protocol.txt", "1")
			}
			write("cmd/tool/main.go", command)
			imports := `"os/exec"`
			invoke := `exec.CommandContext(t.Context(), "go", "run", "../../cmd/tool").CombinedOutput()`
			if change == "shared launcher" {
				write("internal/launcher/launcher.go", "package launcher\nimport (\"context\"; \"os/exec\")\nfunc Run(ctx context.Context) ([]byte,error) { return exec.CommandContext(ctx, \"go\", \"run\", \"../../cmd/tool\").CombinedOutput() }\n")
				imports, invoke = `"overgo/internal/launcher"`, `launcher.Run(t.Context())`
			}
			write("internal/subprocess/wire_test.go", "package subprocess\nimport (\"testing\"; "+imports+")\nfunc TestWire(t *testing.T) { out,err := "+invoke+"; if err != nil || string(out) != \"1\" { t.Fatalf(\"wire=%s err=%v\",out,err) } }\n")
			runGitFixture(t, g.repo, "add", ".")
			const consumer, independent = "overgo/internal/subprocess", "overgo/internal/client"
			started := time.Now()
			if report, err := runGoTests(t.Context(), g.repo, []string{consumer, independent}, false, nil); err != nil {
				t.Fatalf("baseline failed: %+v, %v", report, err)
			}
			graph, err := g.inputGraph()
			if err != nil {
				t.Fatal(err)
			}
			before, err := packageInputIdentities(graph, []string{consumer, independent})
			if err != nil {
				t.Fatal(err)
			}
			cache := automationcheck.NewEvidenceCache(lifecycleTestEnvironment(t).ID)
			for pkg, input := range before {
				if err := cache.RecordPackagePass(pkg, "complete", input); err != nil {
					t.Fatal(err)
				}
			}
			g.paths = []string{"cmd/tool/main.go"}
			switch change {
			case "deleted command":
				if err := os.Remove(filepath.Join(g.repo, "cmd/tool/main.go")); err != nil {
					t.Fatal(err)
				}
			case "renamed command":
				if err := os.Remove(filepath.Join(g.repo, "cmd/tool/main.go")); err != nil {
					t.Fatal(err)
				}
				write("cmd/tool/renamed.go", strings.Replace(command, "Print(1)", "Print(2)", 1))
				g.paths = append(g.paths, "cmd/tool/renamed.go")
			case "unowned protocol file":
				write("protocol.txt", "2")
				g.paths = []string{"protocol.txt"}
			default:
				write("cmd/tool/main.go", strings.Replace(command, "Print(1)", "Print(2)", 1))
			}
			g.packageGraph = nil
			full, fullErr := runGoTests(t.Context(), g.repo, fixtureRootPackages(t, g), false, nil)
			if fullErr == nil || !slices.Contains(full.Failed, consumer) {
				t.Fatalf("seeded regression did not fail full checks: %+v, %v", full, fullErr)
			}
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			selected := append(slices.Clone(scope.direct), scope.dependent...)
			if missed := selectionCounterexamples(full.Failed, selected); len(missed) != 0 {
				t.Errorf("subprocess regression omitted: %v; selected=%v", missed, selected)
			}
			graph, err = g.inputGraph()
			if err != nil {
				t.Fatal(err)
			}
			after, err := packageInputIdentities(graph, []string{consumer, independent})
			if err != nil {
				t.Fatal(err)
			}
			if reused, err := cache.PackageReusable(consumer, "complete", after[consumer]); err != nil || reused {
				t.Errorf("stale subprocess receipt reusable=%t error=%v", reused, err)
			}
			if reused, err := cache.PackageReusable(independent, "complete", after[independent]); err != nil || !reused {
				t.Errorf("independent receipt lost: reusable=%t error=%v", reused, err)
			}
			if t.Failed() {
				return
			}
			report, err := runGoTests(t.Context(), g.repo, selected, false, nil)
			if err == nil || !slices.Equal(report.Failed, full.Failed) {
				t.Fatalf("selected checks disagree with full failures: %v versus %v, %v", report.Failed, full.Failed, err)
			}
			if len(scope.opaqueRuntimeInputs) == 0 {
				t.Fatal("selection omitted subprocess explanation")
			}
			t.Logf("case=%s baseline_passes=2 full_failure_entries=%d selected=%d excluded=%d stale_receipts=0 independent_reused=1 wall=%s", change, len(full.Failed), len(selected), scope.excluded, time.Since(started))
		})
	}
}
