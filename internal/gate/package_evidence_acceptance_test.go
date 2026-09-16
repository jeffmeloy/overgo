package gate

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/repoanalysis"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
)

func TestIndependentPackageEvidenceAcceptance(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	t.Run("real failed group survives restart and retries only failures", func(t *testing.T) {
		root, _ := packageIdentityFixture(t)
		if err := os.Mkdir(filepath.Join(root, "tmp"), 0o755); err != nil {
			t.Fatal(err)
		}
		files := map[string]string{
			"app/app_test.go":          "package app\nimport \"testing\"\nfunc TestRequired(t *testing.T) { t.Fatal(\"declared failure\") }\n",
			"other/other.go":           "package other\nconst Value = 1\n",
			"other/other_test.go":      "package other\nimport (_ \"embed\"; \"testing\")\n//go:embed testdata/value.txt\nvar data string\nfunc TestRequired(t *testing.T) { if data != \"pass\" || Value != 1 { t.Fatalf(\"fixture=%s value=%d\", data, Value) } }\nfunc TestIntegration(t *testing.T) { if testing.Short() { t.Skip(\"" + testskip.ShortIntegration + "\") }; if Value != 1 { t.Fatal(Value) } }\n",
			"other/testdata/value.txt": "pass",
		}
		for name, data := range files {
			path := filepath.Join(root, filepath.FromSlash(name))
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := command(root, "git", "init", "--quiet"); err != nil {
			t.Fatal(err)
		}
		environment, err := discoverEnvironment(root)
		if err != nil {
			t.Fatal(err)
		}
		newGate := func() *gateContext {
			snapshot, err := repoanalysis.DiscoverGo(root, "app", "dep", "other")
			if err != nil {
				t.Fatal(err)
			}
			g := &gateContext{repo: root, storePath: StorePath, environment: environment, paths: []string{"app/app_test.go", "other/other_test.go"}, source: &snapshot}
			t.Cleanup(func() { _ = g.closeStore() })
			return g
		}
		g := newGate()
		if _, err := g.stepTest(t.Context()); err == nil || !strings.Contains(err.Error(), "declared failure") {
			t.Fatalf("group failure lost: %v", err)
		}
		if !slices.Contains(g.audit, "short-profile exclusions (no full-test credit): 1 [example/other: TestIntegration]") {
			t.Fatalf("profile denominator lost: %v", g.audit)
		}
		if err := g.closeStore(); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(filepath.Join(root, filepath.FromSlash(gateRetryFile))); err != nil {
			t.Fatal(err)
		}
		g = newGate()
		graph, err := g.inputGraph()
		if err != nil {
			t.Fatal(err)
		}
		packages := []string{"example/app", "example/other"}
		inputs, err := packageInputIdentities(graph, packages)
		if err != nil {
			t.Fatal(err)
		}
		ledger, err := g.openPackageEvidence()
		if err != nil {
			t.Fatal(err)
		}
		if err := ledger.prepare(t.Context(), packages, "short", inputs, g.retryCache); err != nil {
			t.Fatal(err)
		}
		pending, reused, err := g.packageCachePartition(packages, "short", inputs)
		if err != nil || reused != 1 || !slices.Equal(pending, []string{"example/app"}) {
			t.Fatalf("restart partition pending=%v reused=%d error=%v", pending, reused, err)
		}
		if _, err := g.stepTest(t.Context()); err == nil || !strings.Contains(err.Error(), "declared failure") {
			t.Fatalf("retry failure lost: %v", err)
		}
		if !slices.Contains(g.audit, "package test work: 1 reused profiles; 1 started attempts") {
			t.Fatalf("retry denominator missing: %v", g.audit)
		}
		if err := os.WriteFile(filepath.Join(root, "app", "app_test.go"), []byte("package app\nimport \"testing\"\nfunc TestRequired(t *testing.T) { t.Log(\"repaired\") }\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		g = newGate()
		if _, err := g.stepTest(t.Context()); err != nil {
			t.Fatal(err)
		}
		if !slices.Contains(g.audit, "package test work: 1 reused profiles; 1 started attempts") {
			t.Fatalf("repair denominator missing: %v", g.audit)
		}
		g = newGate()
		if _, err := g.stepTest(t.Context()); err != nil {
			t.Fatal(err)
		}
		if !slices.Contains(g.audit, "package test work: 2 reused profiles; 0 started attempts") {
			t.Fatalf("complete reuse denominator missing: %v", g.audit)
		}
		for _, tc := range []struct{ path, target string }{
			{"dep/dep.go", "example/app"},
			{"other/testdata/value.txt", "example/other"},
		} {
			path := filepath.Join(root, filepath.FromSlash(tc.path))
			original, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, append(slices.Clone(original), '\n'), 0o644); err != nil {
				t.Fatal(err)
			}
			inputs, err := packageInputIdentities(graph, packages)
			if err != nil {
				t.Fatal(err)
			}
			pending, reused, err := newGate().packageCachePartition(packages, "short", inputs)
			if err != nil || reused != 1 || !slices.Equal(pending, []string{tc.target}) {
				t.Fatalf("changed %s: pending=%v reused=%d error=%v", tc.path, pending, reused, err)
			}
			if err := os.WriteFile(path, original, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		inputs, err = packageInputIdentities(graph, packages)
		if err != nil {
			t.Fatal(err)
		}
		if pending, reused, err := newGate().packageCachePartition(packages, "complete", inputs); err != nil || reused != 0 || !slices.Equal(pending, packages) {
			t.Fatalf("short evidence crossed mode: pending=%v reused=%d error=%v", pending, reused, err)
		}
		t.Run("environment and zero matching tests", func(t *testing.T) {
			t.Setenv("GOFLAGS", strings.TrimSpace(os.Getenv("GOFLAGS")+" -run=^TestAbsent$"))
			g := newGate()
			g.environment, err = discoverEnvironment(root)
			if err != nil {
				t.Fatal(err)
			}
			if g.environment.ID == environment.ID {
				t.Fatal("GOFLAGS did not change environment identity")
			}
			if pending, reused, err := g.packageCachePartition(packages, "short", inputs); err != nil || reused != 0 || !slices.Equal(pending, packages) {
				t.Fatalf("environment reused stale evidence: %v %d %v", pending, reused, err)
			}
			report, err := (&gateContext{repo: root}).runGoTests(t.Context(), packages, true, nil)
			if err != nil {
				t.Fatal(err)
			}
			if report.PassedTests != 0 || report.PackagePassed("example/app") || report.PackagePassed("example/other") {
				t.Fatalf("zero-match evidence credited: %+v", report)
			}
		})
		if err := os.WriteFile(filepath.Join(root, "other", "other.go"), []byte("package other\nimport \"example/dep\"\nconst Value = dep.Value\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "app", "app_test.go"), []byte("package app\nimport \"testing\"\nfunc TestRequired(t *testing.T) { t.Skip(\"missing fixture\") }\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		for attempt := range 2 {
			g := newGate()
			g.paths = []string{"dep/dep.go"}
			if _, err := g.stepTest(t.Context()); err != nil {
				t.Fatal(err)
			}
			want := "package test work: 0 reused profiles; 3 started attempts"
			if attempt != 0 {
				want = "package test work: 1 reused profiles; 2 started attempts"
			}
			if !slices.Contains(g.audit, want) || !slices.Contains(g.audit, "dependent fixture evidence not credited: 1 skipped [example/app: TestRequired], 0 unavailable []") {
				t.Fatalf("dependent skip/reuse denominator: %v", g.audit)
			}
		}
	})
	t.Run("partial and invalid streams", assertIndependentPackageStreams)
	t.Run("old aggregate evidence is not reusable", func(t *testing.T) {
		root, graph := packageIdentityFixture(t)
		input, err := graph.identity("example/app")
		if err != nil {
			t.Fatal(err)
		}
		environment, err := artifact.IdentifyBytes(artifact.KindProfile, []byte(root))
		if err != nil {
			t.Fatal(err)
		}
		cache := automationcheck.NewEvidenceCache(environment)
		if err := cache.RecordPackagePass("example/app", "short", input); err != nil {
			t.Fatal(err)
		}
		old, err := artifact.JSONID(artifact.KindRecipe, struct {
			Package string `json:"package"`
			Mode    string `json:"mode"`
		}{"example/app", "short"})
		if err != nil {
			t.Fatal(err)
		}
		for key, entry := range cache.Entries {
			delete(cache.Entries, key)
			entry.Invocation = old
			cache.Entries[old.String()] = entry
			break
		}
		if hit, err := cache.PackageReusable("example/app", "short", input); err != nil || hit {
			t.Fatalf("old aggregate evidence reused: %t %v", hit, err)
		}
	})
}

func assertIndependentPackageStreams(t *testing.T) {
	event := func(action, pkg, test, output string) string {
		data, err := json.Marshal(struct{ Action, Package, Test, Output string }{action, pkg, test, output})
		if err != nil {
			t.Fatal(err)
		}
		return string(data) + "\n"
	}
	start := func(pkg string) string { return event("start", pkg, "", "") + event("run", pkg, "TestRequired", "") }
	pass := func(pkg string) string { return event("pass", pkg, "TestRequired", "") + event("pass", pkg, "", "") }
	good := start("good") + pass("good")
	for _, tc := range []struct {
		name, tail string
		invalid    bool
	}{
		{"failed sibling", start("bad") + event("fail", "bad", "TestRequired", "") + event("fail", "bad", "", ""), false},
		{"skipped sibling", start("bad") + event("skip", "bad", "TestRequired", "") + event("pass", "bad", "", ""), false},
		{"classified skip", start("bad") + event("output", "bad", "TestRequired", testskip.ShortIntegration) + event("skip", "bad", "TestRequired", "") + event("pass", "bad", "", ""), false},
		{"unavailable test", start("bad") + event("output", "bad", "TestRequired", "UNAVAILABLE") + pass("bad"), false},
		{"unavailable package", start("bad") + event("output", "bad", "", "UNAVAILABLE") + pass("bad"), false},
		{"unfinished package", start("bad") + event("pass", "bad", "TestRequired", ""), false},
		{"unfinished subtest", start("bad") + event("run", "bad", "TestRequired/child", "") + pass("bad"), false},
		{"failed subtest", start("bad") + event("run", "bad", "TestRequired/child", "") + event("fail", "bad", "TestRequired/child", "") + pass("bad"), false},
		{"no test files", event("start", "bad", "", "") + event("skip", "bad", "", ""), false},
		{"no matching tests", event("start", "bad", "", "") + event("pass", "bad", "", ""), false},
		{"missing starts", pass("bad"), false},
		{"duplicate terminal", start("bad") + pass("bad") + event("pass", "bad", "", ""), false},
		{"test after package terminal", start("bad") + event("pass", "bad", "", "") + event("pass", "bad", "TestRequired", ""), false},
		{"failed then passed", start("bad") + event("fail", "bad", "TestRequired", "") + pass("bad"), false},
		{"malformed stream", "{broken\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report, err := testevidence.GoTestJSONShortReport(good + tc.tail)
			if (err != nil) != tc.invalid {
				t.Fatalf("parse error: %v", err)
			}
			if report.PackagePassed("good") == tc.invalid || report.PackagePassed("bad") || report.PackagePassed("absent") {
				t.Fatalf("incorrect package credit: good=%t bad=%t report=%+v", report.PackagePassed("good"), report.PackagePassed("bad"), report)
			}
		})
	}
	for _, output := range []string{"", start("bad") + event("skip", "bad", "TestRequired", "") + event("pass", "bad", "", "")} {
		report, _ := testevidence.GoTestJSONShortReport(output)
		if report.PackagePassed("bad") {
			t.Fatal("empty or all-skipped stream credited")
		}
	}
}
