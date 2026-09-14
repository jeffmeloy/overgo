package gate

import (
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

// TestSelectionCauseHistogram reconstructs counts and causal paths from the
// retained gate result and selection-cause record: a package with several
// sufficient causes keeps every one under a single primary label, an
// early-failed check leaves its unobserved packages unstarted, and the
// gate's own retention assembles the record from its execution batches.
func TestSelectionCauseHistogram(t *testing.T) {
	t.Parallel()
	recipeID, err := artifact.JSONID(artifact.KindRecipe, "selection cause fixture recipe")
	if err != nil {
		t.Fatal(err)
	}
	environmentID, err := artifact.JSONID(artifact.KindEvidence, "selection cause fixture environment")
	if err != nil {
		t.Fatal(err)
	}
	gateRecord, err := runrecord.NewGateRecord(recipeID, environmentID, strings.Repeat("a", 40), runrecord.OutcomeFailed, "test-device", uint64(3*time.Minute), []runrecord.GateStep{
		{Name: "scope", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepSucceeded, DurationNS: uint64(time.Second)},
		{Name: "test-owners", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: uint64(time.Minute)},
		{Name: "test-device", Phase: runrecord.PhaseTest, Outcome: runrecord.StepFailed, DurationNS: uint64(2 * time.Minute)},
		{Name: "test", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSkipped},
	})
	if err != nil {
		t.Fatal(err)
	}
	result := gateRecord.Result
	resultID := result.ID
	elapsed := func(seconds float64) *float64 { return new(seconds) }
	record, err := runrecord.SelectionCauseCodec.NewInitial(runrecord.SelectionCauseRecord{
		Result:  resultID,
		Changed: []string{"internal/reader/reader.go", "docs/config.txt"},
		Packages: []runrecord.SelectionPackage{
			// Two sufficient causes: a compiled change and a named runtime input.
			{Package: "overgo/internal/readerclient", Step: "test-owners", Action: "pass", Started: true, ElapsedSeconds: elapsed(4),
				CompilerInputs: []string{"internal/reader/reader.go"}, RuntimeInputs: []string{"docs/config.txt"},
				RuntimeReaders: map[string]string{"overgo/internal/reader": "reads a path the source does not name"}},
			// Only the opaque reader binds this one to every root.
			{Package: "overgo/internal/opaque", Step: "test-owners", Action: "pass", Started: true, ElapsedSeconds: elapsed(1.5),
				UnboundInputs:  []string{"docs/config.txt", "internal/reader/reader.go"},
				RuntimeReaders: map[string]string{"overgo/internal/opaque": "reads every repository input"}},
			// Selected as a dependent with no changed input of its own.
			{Package: "overgo/internal/pure", Step: "test-owners", Action: "pass", Started: true, ElapsedSeconds: elapsed(0.5),
				UnboundInputs: []string{"docs/config.txt", "internal/reader/reader.go"}},
			// The device check failed on its first package; the second never started.
			{Package: "overgo/internal/cuda/kernel", Step: "test-device", Action: "fail", Started: true, ElapsedSeconds: elapsed(90),
				CompilerInputs: []string{"internal/reader/reader.go"}},
			{Package: "overgo/internal/cuda/executor", Step: "test-device",
				CompilerInputs: []string{"internal/reader/reader.go"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(record.Changed, []string{"docs/config.txt", "internal/reader/reader.go"}) {
		t.Fatalf("changed inputs not canonical: %v", record.Changed)
	}
	histogram, err := runrecord.SelectionCauseHistogram(result, record)
	if err != nil {
		t.Fatal(err)
	}
	steps := map[string]runrecord.SelectionStepSummary{}
	for _, step := range histogram.Steps {
		steps[step.Name] = step
	}
	if len(histogram.Steps) != 2 || steps["test-owners"].Packages != 3 || steps["test-owners"].Executed != 3 ||
		steps["test-device"].Packages != 2 || steps["test-device"].Executed != 1 || steps["test-device"].Failed != 1 || steps["test-device"].Unstarted != 1 {
		t.Fatalf("step partition = %+v", histogram.Steps)
	}
	primary := map[string]runrecord.SelectionCauseCount{}
	for _, bar := range histogram.Primary {
		primary[bar.Cause] = bar
	}
	if primary[runrecord.SelectionCauseCompiler].Packages != 3 || primary[runrecord.SelectionCauseCompiler].Executed != 2 ||
		primary[runrecord.SelectionCauseCompiler].Unstarted != 1 || primary[runrecord.SelectionCauseCompiler].ElapsedSeconds != 94 ||
		primary[runrecord.SelectionCauseReader].Packages != 1 || primary[runrecord.SelectionCauseDependent].Packages != 1 ||
		len(primary) != 3 {
		t.Fatalf("primary histogram = %+v", histogram.Primary)
	}
	sufficient := map[string]int{}
	for _, bar := range histogram.Sufficient {
		sufficient[bar.Cause] = bar.Packages
	}
	// The reader client counts under compiler, runtime and opaque-reader at once.
	if sufficient[runrecord.SelectionCauseCompiler] != 3 || sufficient[runrecord.SelectionCauseRuntime] != 1 ||
		sufficient[runrecord.SelectionCauseReader] != 2 || sufficient[runrecord.SelectionCauseDependent] != 1 {
		t.Fatalf("sufficient causes = %+v", histogram.Sufficient)
	}
	if len(histogram.Inputs) != 2 || histogram.Inputs[0].Cause != "internal/reader/reader.go" || histogram.Inputs[0].Packages != 3 ||
		histogram.Inputs[1].Cause != "docs/config.txt" || histogram.Inputs[1].Packages != 1 {
		t.Fatalf("input ranking = %+v", histogram.Inputs)
	}
	var client runrecord.SelectionPackageCauses
	for _, entry := range histogram.Packages {
		if entry.Package == "overgo/internal/readerclient" {
			client = entry
		}
	}
	if client.Primary != runrecord.SelectionCauseCompiler || len(client.Causes) != 3 || !client.Executed ||
		client.Causes[1].Kind != runrecord.SelectionCauseRuntime || !strings.Contains(client.Causes[2].Detail, "reads a path") {
		t.Fatalf("causal path = %+v", client)
	}
	// Cost is attributed only to observed execution.
	for _, entry := range histogram.Packages {
		if entry.Package == "overgo/internal/cuda/executor" && (entry.Executed || entry.ElapsedSeconds != nil) {
			t.Fatalf("unstarted package attributed cost: %+v", entry)
		}
	}
	// A record for another result never explains this one.
	other := record
	other.Result = artifact.ID{}
	if _, err := runrecord.SelectionCauseHistogram(result, other); err == nil {
		t.Fatal("foreign selection record accepted")
	}
	mismatch := record
	mismatch.Packages = append(slices.Clone(record.Packages), runrecord.SelectionPackage{Package: "overgo/internal/x", Step: "absent"})
	if _, err := runrecord.SelectionCauseHistogram(result, mismatch); err == nil {
		t.Fatal("package bound to an absent step accepted")
	}

	// The gate assembles the record from its recorded batches: a retried
	// package keeps its observed execution, an unobserved one stays unstarted.
	g := runtimeReaderFixture(t)
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	const target = "overgo/internal/readerclient"
	const unstarted = "overgo/internal/isolated"
	g.paths = []string{"docs/config.txt"}
	g.testPlan = &testGroups{edited: []string{target, unstarted}, directInputs: map[string]artifact.ID{}}
	g.setTestStep(testOwnersCheckName)
	g.recordPackageExecution([]string{target, unstarted}, true, time.Second, testevidence.GoTestReport{
		Executions: []testevidence.PackageExecution{{Package: target, Action: "fail", Started: true, Elapsed: elapsed(0.5)}},
	}, nil)
	g.recordPackageExecution([]string{target}, true, time.Second, testevidence.GoTestReport{
		Executions: []testevidence.PackageExecution{{Package: target, Action: "pass", Started: true, Elapsed: elapsed(0.7)}},
	}, nil)
	g.retainSelectionCauses(graph.normalizePackageExecutions(g.testExecutions))
	if len(g.selectionCauses) != 2 {
		t.Fatalf("retained packages = %+v", g.selectionCauses)
	}
	for _, entry := range g.selectionCauses {
		switch entry.Package {
		case target:
			if entry.Action != "pass" || *entry.ElapsedSeconds != 0.7 || !slices.Equal(entry.RuntimeInputs, g.paths) || entry.Step != testOwnersCheckName {
				t.Fatalf("retry lost its observed execution: %+v", entry)
			}
		case unstarted:
			if entry.Started || entry.Action != "" || entry.ElapsedSeconds != nil {
				t.Fatalf("unobserved package attributed execution: %+v", entry)
			}
		default:
			t.Fatalf("unexpected package %s", entry.Package)
		}
	}
	var batch artifact.Batch
	final := resultID
	if err := g.appendSelectionCauses(&batch, final); err != nil {
		t.Fatal(err)
	}
	if len(batch.Contents) != 1 || len(batch.Lineage) != 1 || batch.Lineage[0].Parent != final {
		t.Fatalf("selection record not bound to the gate result: %+v", batch)
	}
	retained, err := runrecord.SelectionCauseCodec.Parse(batch.Contents[0].Data)
	if err != nil || retained.Result != final || len(retained.Packages) != 2 {
		t.Fatalf("retained record = %+v, %v", retained, err)
	}
}
