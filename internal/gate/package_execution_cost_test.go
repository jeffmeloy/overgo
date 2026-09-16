package gate

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestGatePackageExecutionCostAcceptance(t *testing.T) {
	t.Parallel()
	g := runtimeReaderFixture(t)
	reader := filepath.Join(g.repo, "internal/reader/reader.go")
	source, err := os.ReadFile(reader)
	if err != nil {
		t.Fatal(err)
	}
	source = append(source, []byte("\nfunc Scan(path string) ([]byte,error) { return os.ReadFile(path) }\n")...)
	testutil.WriteTextFile(t, g.repo, "internal/reader/reader.go", string(source), 0600)
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	const target = "overgo/internal/readerclient"
	const unstarted = "overgo/internal/isolated"
	g.paths = []string{"docs/config.txt"}
	g.testPlan = &testGroups{edited: []string{target, unstarted, "overgo/internal/reader"}}
	g.setTestStep(testOwnersCheckName)
	report, err := g.runGoTestsAdmitted(t.Context(), []string{target}, true, nil, false)
	if err != nil || !report.PackagePassed(target) {
		t.Fatalf("live execution: %v, %+v", err, report)
	}
	if len(g.testExecutions) != 1 || len(g.testExecutions[0].Executions) != 1 || g.testExecutions[0].Executions[0].Elapsed == nil {
		t.Fatalf("runner lost package timing: %+v", g.testExecutions)
	}
	// Preserve an independent failed attempt; planned work supplies no timing.
	g.recordPackageExecution([]string{target, unstarted}, false, time.Second, testevidence.GoTestReport{
		Executions: []testevidence.PackageExecution{{Package: target, Action: "fail", Started: true, Elapsed: new(0.5)}},
	}, errors.New("seeded failure"))
	g.captureSelectionCauses()
	if len(g.testExecutions) != 2 || !g.testExecutions[0].Short || g.testExecutions[1].Short || !g.testExecutions[1].Failed {
		t.Fatalf("profiles or attempts coalesced: %+v", g.testExecutions)
	}
	failed := g.testExecutions[1]
	if !slices.Equal(failed.Unobserved, []string{unstarted}) || *failed.Executions[0].Elapsed != 0.5 || failed.WallNS != uint64(time.Second) {
		t.Fatalf("planned or phase cost attributed to execution: %+v", failed)
	}
	readers := false
	for _, entry := range g.selectionCauses {
		for reader, reason := range entry.RuntimeReaders {
			readers = true
			if reason == "" {
				t.Fatalf("unbound reader %s", reader)
			}
		}
	}
	if !readers {
		t.Fatal("reader diagnostic missing")
	}
	if len(g.audit) != 0 {
		t.Fatalf("execution data copied into advisories: %v", g.audit)
	}
	before, err := graph.identity(target)
	if err != nil {
		t.Fatal(err)
	}
	// Relative selectors resolve to the same observed package, not unstarted work.
	batch := graph.normalizePackageExecutions([]packageExecutionBatch{{Requested: []string{"./internal/readerclient"}, Executions: report.Executions}})
	if len(batch[0].Unobserved) != 0 || !slices.Equal(batch[0].Requested, []string{target}) {
		t.Fatalf("selector resolution: %+v", batch)
	}
	g.packageExecutionAudit()
	after, err := graph.identity(target)
	if err != nil || before != after {
		t.Fatalf("cost reporting invalidated receipt: %v", err)
	}
}

func TestPackageExecutionAuditUsesObservedWork(t *testing.T) {
	t.Parallel()
	g := &gateContext{testPlan: &testGroups{reused: 1, edited: []string{"changed", "build-failed"}, remaining: []string{"host", "device"}, dependent: []string{"changed"}}, deferLanes: true,
		packageGraph: &packageInputGraph{nodes: []goPackageInput{{ImportPath: "device", TestImports: []string{deviceTestPackages[0]}}}},
	}
	g.recordPackageExecution([]string{"changed", "build-failed"}, true, time.Second, testevidence.GoTestReport{Executions: []testevidence.PackageExecution{
		{Package: "changed", Started: true, Action: "fail"}, {Package: "build-failed", Action: "fail"},
	}}, errors.New("failure"))
	g.recordPackageExecution([]string{"changed"}, true, time.Second, testevidence.GoTestReport{Executions: []testevidence.PackageExecution{{Package: "changed", Started: true, Action: "pass"}}}, nil)
	g.packageExecutionAudit()
	if !slices.Contains(g.audit, "package test work: 1 reused profiles; 2 started attempts") || !slices.Contains(g.audit, "package observations: passed=1 failed=2 skipped=0 interrupted=0 unstarted=1; awaiting=3 deferred=1; observations grant no evidence credit") {
		t.Fatalf("planned work, profiles or failed retry collapsed: %v", g.audit)
	}
	g.recordPackageExecution([]string{"host", "device"}, true, time.Second, testevidence.GoTestReport{Executions: []testevidence.PackageExecution{{Package: "host", Started: true, Action: "skip"}, {Package: "device", Started: true}}}, errors.New("interrupted"))
	g.packageExecutionAudit()
	if !slices.Contains(g.audit, "package observations: passed=1 failed=2 skipped=1 interrupted=1 unstarted=1; awaiting=1 deferred=0; observations grant no evidence credit") {
		t.Fatalf("skip, interruption or unobserved full profile lost: %v", g.audit)
	}
	if len(g.testExecutions) != 3 || g.testPlan.reused != 1 {
		t.Fatal("diagnostic changed retained execution or reuse authority")
	}
}
