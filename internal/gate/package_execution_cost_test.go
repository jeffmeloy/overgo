package gate

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
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
	if err := os.WriteFile(reader, source, 0600); err != nil {
		t.Fatal(err)
	}
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
	g.dependencyCostAudit([]runrecord.GateStep{{Name: "test-owners", Outcome: runrecord.StepFailed, DurationNS: uint64(time.Minute)}})
	var audit struct {
		DurationNS uint64                   `json:"duration_ns"`
		Executions []packageExecutionBatch  `json:"executions"`
		Packages   []packageCostAttribution `json:"packages"`
		Readers    map[string]string        `json:"readers"`
	}
	line := g.audit[len(g.audit)-1]
	if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "test input attribution: ")), &audit); err != nil {
		t.Fatal(err)
	}
	if len(audit.Executions) != 2 || !audit.Executions[0].Short || audit.Executions[1].Short || !audit.Executions[1].Failed {
		t.Fatalf("profiles or attempts coalesced: %+v", audit.Executions)
	}
	failed := audit.Executions[1]
	if !slices.Equal(failed.Unobserved, []string{unstarted}) || *failed.Executions[0].Elapsed != 0.5 || failed.WallNS != uint64(time.Second) || audit.DurationNS != uint64(time.Minute) {
		t.Fatalf("planned or phase cost attributed to execution: %+v", failed)
	}
	for _, entry := range audit.Packages {
		for _, reader := range entry.RuntimeReaders {
			if audit.Readers[reader] == "" {
				t.Fatalf("unbound reader %s", reader)
			}
		}
	}
	if len(audit.Readers) == 0 {
		t.Fatal("reader diagnostic missing")
	}
	for _, reason := range audit.Readers {
		quoted, err := json.Marshal(reason)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Count(line, string(quoted)) != 1 {
			t.Fatal("reader explanation duplicated per consumer")
		}
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
	after, err := graph.identity(target)
	if err != nil || before != after {
		t.Fatalf("cost reporting invalidated receipt: %v", err)
	}
}
