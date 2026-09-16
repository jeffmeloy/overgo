package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/repoanalysis"
)

func TestGateScopePreservesAffectedCoverage(t *testing.T) {
	t.Parallel()
	g := scopeCompilerFixture(t)
	imports := func(names ...string) []string {
		var result []string
		for _, name := range names {
			result = append(result, "overgo/internal/"+name)
		}
		slices.Sort(result)
		return result
	}
	productionDependents := imports("client", "consumer", "testonly")
	for _, tc := range []struct {
		name              string
		paths             []string
		direct, dependent []string
		assembly          bool
	}{
		{"shared owner test", []string{"internal/plan/plan_test.go"}, imports("plan"), nil, false},
		{"test import owner", []string{"internal/testonly/check_test.go"}, imports("testonly"), nil, false},
		{"production and test importers", []string{"internal/plan/plan.go"}, imports("plan"), productionDependents, true},
		{"dependency rebuilt for another test target", []string{"internal/consumer/consumer.go"}, imports("consumer"), imports("client", "plan"), false},
		{"production embed", []string{"internal/plan/catalog.txt"}, imports("plan"), productionDependents, true},
		{"compiled assembly", []string{"internal/plan/helpers.s"}, imports("plan"), productionDependents, true},
		{"compiled header", []string{"internal/plan/helpers.h"}, imports("plan"), productionDependents, true},
		{"removed asset", []string{"internal/plan/testdata/removed.txt"}, imports("plan"), productionDependents, true},
		{"test embed", []string{"internal/plan/testdata/fixture.txt"}, imports("plan"), nil, false},
		{"undeclared asset stays conservative", []string{"internal/plan/testdata/runtime.txt"}, imports("plan"), productionDependents, true},
		{"test source embedded in production", []string{"internal/plan/payload_test.go"}, imports("plan"), productionDependents, true},
		{"deleted test", []string{"internal/plan/deleted_test.go"}, imports("plan"), nil, false},
		{"cross owner rename", []string{"internal/plan/old.go", "internal/other/new.go"}, imports("plan", "other"), productionDependents, true},
		{"removed package widens", []string{"internal/removed/old.go"}, nil, imports("client", "consumer", "other", "plan", "recipe", "testonly", "testonlyclient"), true},
		{"module input widens", []string{"go.mod"}, nil, imports("client", "consumer", "other", "plan", "recipe", "testonly", "testonlyclient"), true},
		{"plan and compatibility authority", []string{"docs/plan.json", "compatibility.json"}, nil, nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g.paths = tc.paths
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(scope.direct, tc.direct) || !slices.Equal(scope.dependent, tc.dependent) {
				t.Errorf("scope direct=%v dependent=%v; want %v / %v", scope.direct, scope.dependent, tc.direct, tc.dependent)
			}
			coverage, err := automationcheck.AgentHarnessBoundaryCoverage(*g.source, scope.productionPaths)
			if err != nil {
				t.Fatal(err)
			}
			if (len(coverage.Boundaries) != 0) != tc.assembly {
				t.Errorf("production assembly scope = %+v", coverage)
			}
			selected := scope.selected()
			if err := automationcheck.RequireAgentHarnessBoundaries(coverage, selected); err != nil {
				t.Fatal(err)
			}
			if tc.assembly && automationcheck.RequireAgentHarnessBoundaries(coverage, scope.direct) == nil {
				t.Fatal("missing assembled consumers accepted")
			}
		})
	}
	if phaseReusesEvidence("acceptance") || !phaseOwnsPath("claims", "compatibility.json") || !phaseOwnsPath("docs", "docs/plan.json") {
		t.Fatal("scope narrowing relaxed plan or compatibility authority")
	}
	t.Run("compiled input evidence cannot go stale", func(t *testing.T) { assertScopeInputEvidence(t, g) })
	t.Run("measured baseline gate incident", assertScopeMeasuredIncident)
	t.Run("real downstream failure is retained", func(t *testing.T) {
		path := filepath.Join(g.repo, "internal", "plan", "plan.go")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(strings.Replace(string(data), "Value = 1", "Value = 2", 1)), 0o644); err != nil {
			t.Fatal(err)
		}
		g.paths = []string{"internal/plan/plan.go"}
		scope, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		report, err := g.runGoTests(t.Context(), scope.dependent, false, nil)
		if err == nil {
			t.Fatal("production regression passed selected consumers")
		}
		for _, owner := range productionDependents {
			if !slices.Contains(report.Failed, owner) {
				t.Errorf("affected consumer %s failure omitted: %v", owner, report.Failed)
			}
		}
	})
}

func assertScopeInputEvidence(t *testing.T, g *gateContext) {
	t.Helper()
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	environment, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("scope fixture environment"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ path, target string }{
		{"internal/plan/testdata/fixture.txt", "overgo/internal/plan"},
		{"internal/plan/testdata/runtime.txt", "overgo/internal/client"},
		{"internal/plan/catalog.txt", "overgo/internal/client"},
		{"internal/plan/helpers.s", "overgo/internal/client"},
		{"internal/plan/helpers.h", "overgo/internal/client"},
		{"internal/plan/payload_test.go", "overgo/internal/client"},
		{"go.mod", "overgo/internal/other"},
	} {
		before, err := graph.identity(tc.target)
		if err != nil {
			t.Fatal(err)
		}
		cache := automationcheck.NewEvidenceCache(environment)
		if err := cache.RecordPackagePass(tc.target, "complete", before); err != nil {
			t.Fatal(err)
		}
		if reused, err := cache.PackageReusable(tc.target, "complete", before); err != nil || !reused {
			t.Fatalf("unchanged %s lost reusable evidence: %v", tc.target, err)
		}
		path := filepath.Join(g.repo, filepath.FromSlash(tc.path))
		original, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(slices.Clone(original), '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		after, err := graph.identity(tc.target)
		if err != nil {
			t.Fatal(err)
		}
		if reused, err := cache.PackageReusable(tc.target, "complete", after); err != nil || reused {
			t.Fatalf("changed %s reused evidence for %s: %v", tc.path, tc.target, err)
		}
		if err := os.WriteFile(path, original, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(g.repo, "internal", "plan", "testdata", "runtime.txt")
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	missing, err := graph.identity("overgo/internal/client")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	empty, err := graph.identity("overgo/internal/client")
	if err != nil || empty == missing {
		t.Fatalf("missing resource conflated with an empty file: %v", err)
	}
	if err := os.WriteFile(path, original, 0o644); err != nil {
		t.Fatal(err)
	}
}

func assertScopeMeasuredIncident(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	// This actual gated slice took 973.8s and selected 2 direct plus 124
	// downstream packages because its plan change included one test-only edit.
	changed, err := command(root, "git", "diff-tree", "--no-commit-id", "--name-only", "-r", "1939094b")
	if err != nil {
		t.Fatal(err)
	}
	live := liveRepositoryFixture(t)
	g := live.context(strings.Fields(changed)...)
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"overgo/cmd/longform", "overgo/internal/plan"}
	for _, owner := range want {
		if !slices.Contains(scope.direct, owner) {
			t.Fatalf("historical direct owner omitted: %s", owner)
		}
	}
	// The original two-package projection omitted subprocess execution. Its
	// zero-dependent claim is invalid without that independence proof. The
	// compiler fixture above still requires exact scope for ordinary imports.
	if !slices.Contains(scope.opaqueRuntimeInputs, "overgo/internal/gate") {
		t.Fatal("historical projection still omits the gate's opaque subprocess boundary")
	}
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := packageInputIdentities(graph, scope.direct); err != nil {
		t.Fatalf("selected repository inputs cannot establish cache evidence: %v", err)
	}
	legacy, err := command(root, "go", "list", "-f", "{{.ImportPath}} {{join .Deps \",\"}}", "./...")
	if err != nil {
		t.Fatal(err)
	}
	legacyCount := 0
	for line := range strings.SplitSeq(strings.TrimSpace(legacy), "\n") {
		owner, dependencies, _ := strings.Cut(line, " ")
		if slices.Contains(want, owner) || slices.ContainsFunc(strings.Split(dependencies, ","), func(dep string) bool { return slices.Contains(want, dep) }) {
			legacyCount++
		}
	}
	selected := len(scope.direct) + len(scope.dependent)
	t.Logf("incident 1939094b: legacy_import_scope=%d corrected_scope=%d opaque_consumers=%d; historic gate wall=973.8s; prior two-package savings claim reopened, no fresh full-gate throughput claim", legacyCount, selected, len(scope.opaqueRuntimeInputs))
}

func scopeCompilerFixture(t *testing.T) *gateContext {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod":                             "module overgo\n\ngo 1.25\n",
		"internal/plan/plan.go":              "package plan\nimport _ \"embed\"\nconst Value = 1\n//go:embed catalog.txt\nvar Catalog string\n//go:embed payload_test.go\nvar Payload string\n",
		"internal/plan/plan_test.go":         "package plan\nimport (_ \"embed\"; \"testing\")\n//go:embed testdata/fixture.txt\nvar fixture string\nfunc TestValue(t *testing.T) { if Value != 1 { t.Fatal(Value) } }\n",
		"internal/plan/payload_test.go":      "package plan\n",
		"internal/plan/external_test.go":     "package plan_test\nimport (\"testing\"; \"overgo/internal/consumer\")\nfunc TestConsumer(t *testing.T) { if consumer.Value() != 1 { t.Fatal(consumer.Value()) } }\n",
		"internal/plan/catalog.txt":          "production catalog",
		"internal/plan/helpers.s":            "// Assembly input.\n",
		"internal/plan/helpers.h":            "// Header input.\n",
		"internal/plan/testdata/fixture.txt": "test fixture",
		"internal/plan/testdata/runtime.txt": "ordinary file input",
		"internal/recipe/recipe.go":          "package recipe\nconst Value = 0\n",
		"internal/consumer/consumer.go":      "package consumer\nimport (\"overgo/internal/plan\"; \"overgo/internal/recipe\")\nfunc Value() int { return plan.Value + recipe.Value }\n",
		"internal/consumer/consumer_test.go": "package consumer\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value() != 1 { t.Fatal(Value()) } }\n",
		"internal/client/client.go":          "package client\nimport \"overgo/internal/consumer\"\nfunc Value() int { return consumer.Value() }\n",
		"internal/client/client_test.go":     "package client\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value() != 1 { t.Fatal(Value()) } }\n",
		"internal/testonly/library.go":       "package testonly\nconst Value = 0\n",
		"internal/testonly/check_test.go":    "package testonly_test\nimport (\"testing\"; \"overgo/internal/plan\")\nfunc TestValue(t *testing.T) { if plan.Value != 1 { t.Fatal(plan.Value) } }\n",
		"internal/testonlyclient/client.go":  "package testonlyclient\nimport \"overgo/internal/testonly\"\nvar Value = testonly.Value\n",
		"internal/other/other.go":            "package other\n",
	}
	var sources []string
	for name, content := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		if strings.HasSuffix(name, ".go") {
			sources = append(sources, name)
		}
	}
	runGitFixture(t, root, "init", "-q")
	runGitFixture(t, root, "add", ".")
	snapshot, err := repoanalysis.LoadGo(root, sources)
	if err != nil {
		t.Fatal(err)
	}
	return &gateContext{repo: root, source: &snapshot}
}
