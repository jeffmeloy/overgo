package gate

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
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
		if len(g.selectionCauses) != 1 || g.selectionCauses[0].Input != frozen || !slices.Equal(g.selectionCauses[0].RuntimeInputs, g.paths) {
			t.Fatalf("teardown lost frozen attribution: %+v", g.selectionCauses)
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
		testutil.WriteTextFile(t, g.repo, "internal/reader/reader.go", string(source))
		testutil.WriteTextFile(t, g.repo, "internal/pure/pure_test.go", "package pure\nimport (\"testing\";\"overgo/internal/reader\")\nfunc TestPure(t *testing.T) { if reader.Pure()!=1 { t.Fatal(reader.Pure()) } }\n")
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
		testutil.WriteTextFile(t, g.repo, "docs/config.txt", "2\n")
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
		testutil.WriteTextFile(t, g.repo, "internal/reader/reader.go", strings.Replace(string(source), "func Pure() int { return 1 }", "func Pure() int { return 2 }", 1))
		if out, err := command(g.repo, "go", "test", "./internal/pure", "-count=1"); err == nil {
			t.Fatalf("pure implementation mutation escaped its caller: %s", out)
		}
		g.testPlan = &testGroups{edited: []string{pureTarget}}
		g.testExecutions = []packageExecutionBatch{{Step: testOwnersCheckName, Requested: []string{pureTarget}}}
		g.captureSelectionCauses()
		if len(g.selectionCauses) != 1 || g.selectionCauses[0].Step != testOwnersCheckName || g.selectionCauses[0].Package != pureTarget {
			t.Fatalf("attribution includes unrelated work: %+v", g.selectionCauses)
		}
		if len(g.audit) != 0 {
			t.Fatalf("attribution copied into advisories: %v", g.audit)
		}
	})
	t.Run("live device math binding", func(t *testing.T) {
		began := time.Now()
		g := liveGateContext(t)
		graph := *g.packageGraph
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
