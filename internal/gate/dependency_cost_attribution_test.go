package gate

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

func TestDependencyCostAttributionAcceptance(t *testing.T) {
	t.Parallel()
	t.Run("candidate teardown retains attribution", func(t *testing.T) {
		g := runtimeReaderFixture(t)
		runGitFixture(t, g.repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
		g.paths = []string{"docs/config.txt"}
		tree, err := g.plannedTree()
		if err != nil {
			t.Fatal(err)
		}
		var frozen artifact.ID
		err = g.withCandidateWorktree(tree, func(string) error {
			graph, err := g.inputGraph()
			if err != nil {
				return err
			}
			const target = "overgo/internal/readerclient"
			frozen, err = graph.identity(target)
			if err != nil {
				return err
			}
			g.testPlan = &testGroups{edited: []string{target}, directInputs: map[string]artifact.ID{target: frozen}}
			g.testExecutions = []packageExecutionBatch{{Step: testOwnersCheckName, Requested: []string{target}}}
			g.steps = []runrecord.GateStep{{Name: "test-owners", Outcome: runrecord.StepSucceeded, DurationNS: uint64(time.Second)}}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if g.packageGraph != nil || g.candidateRoot != "" {
			t.Fatal("candidate lifetime leaked")
		}
		var report struct {
			Packages []packageCostAttribution `json:"packages"`
		}
		if len(g.audit) != 1 {
			t.Fatalf("attribution lost at candidate teardown: %v", g.audit)
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(g.audit[0], "test input attribution: ")), &report); err != nil {
			t.Fatal(err)
		}
		if len(report.Packages) != 1 || report.Packages[0].Input != frozen || !slices.Equal(report.Packages[0].RuntimeInputs, g.paths) {
			t.Fatalf("teardown lost frozen attribution: %+v", report)
		}
	})
	t.Run("reachable reader and pure sibling", func(t *testing.T) {
		g := runtimeReaderFixture(t)
		reader := filepath.Join(g.repo, "internal", "reader", "reader.go")
		source, err := os.ReadFile(reader)
		if err != nil {
			t.Fatal(err)
		}
		source = append(source, []byte("\nfunc Pure() int { return 1 }\nfunc Scan(path string) ([]byte,error) { return os.ReadFile(path) }\n")...)
		if err := os.WriteFile(reader, source, 0o644); err != nil {
			t.Fatal(err)
		}
		pure := filepath.Join(g.repo, "internal", "pure")
		if err := os.MkdirAll(pure, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(pure, "pure_test.go"), []byte("package pure\nimport (\"testing\";\"overgo/internal/reader\")\nfunc TestPure(t *testing.T) { if reader.Pure()!=1 { t.Fatal(reader.Pure()) } }\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runGitFixture(t, g.repo, "add", ".")
		graph, err := g.inputGraph()
		if err != nil {
			t.Fatal(err)
		}
		const pureTarget = "overgo/internal/pure"
		changes := []string{"docs/config.txt", "internal/reader/reader.go"}
		attribution, err := graph.attributeInputs(pureTarget, changes)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(attribution.CompilerInputs, changes[1:]) || !slices.Equal(attribution.RuntimeInputs, changes[:1]) || len(attribution.UnboundInputs) != 0 {
			t.Fatalf("compiler/runtime distinction = %+v", attribution)
		}
		if !strings.Contains(attribution.RuntimeReaders["overgo/internal/reader"], "reads a path") {
			t.Fatalf("reader boundary missing: %+v", attribution)
		}
		permuted, err := graph.attributeInputs(pureTarget, []string{changes[1], changes[0], changes[1]})
		if err != nil || !reflect.DeepEqual(attribution, permuted) {
			t.Fatalf("unstable attribution = %+v, %v", permuted, err)
		}
		before, err := graph.identity(pureTarget)
		if err != nil {
			t.Fatal(err)
		}
		if out, err := command(g.repo, "go", "test", "./internal/pure", "./internal/readerclient", "-count=1"); err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		g.paths = changes[:1]
		scope, err := g.deriveTestScope()
		if err != nil || !slices.Contains(scope.selected(), pureTarget) {
			t.Fatalf("diagnosis changed conservative selection: %+v, %v", scope, err)
		}
		if err := os.WriteFile(filepath.Join(g.repo, "docs", "config.txt"), []byte("2\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		after, err := graph.identity(pureTarget)
		if err != nil || after == before {
			t.Fatalf("diagnosis changed conservative invalidation: %v", err)
		}
		if out, err := command(g.repo, "go", "test", "./internal/pure", "-count=1"); err != nil {
			t.Fatalf("unrelated runtime mutation affected pure caller: %v\n%s", err, out)
		}
		if out, err := command(g.repo, "go", "test", "./internal/readerclient", "-count=1"); err == nil {
			t.Fatalf("reachable reader failed to detect fixture mutation: %s", out)
		}
		if err := os.WriteFile(reader, []byte(strings.Replace(string(source), "func Pure() int { return 1 }", "func Pure() int { return 2 }", 1)), 0o644); err != nil {
			t.Fatal(err)
		}
		if out, err := command(g.repo, "go", "test", "./internal/pure", "-count=1"); err == nil {
			t.Fatalf("pure implementation mutation escaped its caller: %s", out)
		}
		g.testPlan = &testGroups{edited: []string{pureTarget}}
		g.testExecutions = []packageExecutionBatch{{Step: testOwnersCheckName, Requested: []string{pureTarget}}}
		g.dependencyCostAudit([]runrecord.GateStep{
			{Name: "test-owners", Outcome: runrecord.StepSucceeded, DurationNS: uint64(time.Second)},
			{Name: "test", Outcome: runrecord.StepReused, DurationNS: uint64(2 * time.Second)},
			{Name: "build", Outcome: runrecord.StepSucceeded, DurationNS: uint64(3 * time.Second)},
		})
		if len(g.audit) != 1 || !strings.HasPrefix(g.audit[0], "test input attribution: ") {
			t.Fatalf("missing bounded cost audit: %v", g.audit)
		}
		var audit struct {
			Step       string                   `json:"step"`
			DurationNS uint64                   `json:"duration_ns"`
			Packages   []packageCostAttribution `json:"packages"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(g.audit[0], "test input attribution: ")), &audit); err != nil {
			t.Fatal(err)
		}
		if audit.Step != "test-owners" || audit.DurationNS != uint64(time.Second) || len(audit.Packages) != 1 || audit.Packages[0].Package != pureTarget {
			t.Fatalf("cost scope includes reused checks or unrelated phases: %+v", audit)
		}
		printed := compactAudit(g.audit)
		if len(printed) != 1 || printed[0] != "advisory: dependency: "+g.audit[0] {
			t.Fatalf("typed explanation was dropped or truncated: %v", printed)
		}
	})
	t.Run("live device math binding", func(t *testing.T) {
		live := liveRepositoryFixture(t)
		began := time.Now()
		graph := live.graph
		const target = "overgo/internal/devicemath"
		changes := []string{"internal/evaluation/qwen_retained_text_test.go", "internal/testutil/numeric.go"}
		attribution, err := graph.attributeInputs(target, changes)
		if err != nil {
			t.Fatal(err)
		}
		if slices.Contains(attribution.CompilerInputs, changes[0]) {
			t.Fatalf("foreign test source reported as compiled device input: %+v", attribution)
		}
		before, err := graph.identity(target)
		if err != nil {
			t.Fatal(err)
		}
		again, err := graph.attributeInputs(target, changes)
		if err != nil || !reflect.DeepEqual(attribution, again) {
			t.Fatalf("live explanation drift: %+v, %v", again, err)
		}
		after, err := graph.identity(target)
		if err != nil || before != after {
			t.Fatalf("diagnostic mutated evidence inputs: %v", err)
		}
		// Future call isolation may remove this broad binding. Record the live
		// disposition rather than require today's over-selection forever.
		t.Logf("live target=%s input=%s compiler=%v runtime=%v unbound=%v opaque_compiler_dependencies=%d analysis_wall=%s acquisitions=0", target, before, attribution.CompilerInputs, attribution.RuntimeInputs, attribution.UnboundInputs, len(attribution.RuntimeReaders), time.Since(began))
	})
}
