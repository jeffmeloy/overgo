package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestSurfaceSelectionWitness(t *testing.T) {
	g := runtimeReaderFixture(t)
	g.storePath, g.environment = StorePath, lifecycleTestEnvironment(t)
	t.Cleanup(func() { _ = g.closeStore() })
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	const target = "overgo/internal/readerclient"
	packages := []string{target}
	inputs, err := packageInputIdentities(graph, packages)
	if err != nil {
		t.Fatal(err)
	}
	original := inputs[target]
	ledger, err := g.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	prepare := func(mode string) runrecord.SelectionReuse {
		t.Helper()
		if err := ledger.prepare(t.Context(), packages, mode, inputs, g.retryCache); err != nil {
			t.Fatal(err)
		}
		return ledger.prepared[target]
	}
	missing := prepare("complete")
	if !missing.Obligation.Valid() || missing.Receipt.Valid() || missing.Passed {
		t.Fatalf("initial obligation misreported: %+v", missing)
	}
	if err := ledger.record(t.Context(), target, true); err != nil {
		t.Fatal(err)
	}
	if ledger.prepared[target] != missing {
		t.Fatal("execution rewrote the admission witness")
	}
	passed := prepare("complete")
	if passed.Obligation != missing.Obligation || !passed.Receipt.Valid() || !passed.Passed {
		t.Fatalf("exact prior pass lost: %+v", passed)
	}
	const changed = "docs/config.txt"
	if err := os.WriteFile(filepath.Join(g.repo, filepath.FromSlash(changed)), []byte("2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	inputs[target], err = graph.identity(target)
	if err != nil || inputs[target] == original {
		t.Fatalf("runtime data did not invalidate the consumer input: %v", err)
	}
	changedWitness := prepare("complete")
	if changedWitness.Obligation == passed.Obligation || changedWitness.Receipt.Valid() {
		t.Fatalf("changed input reused old authority: %+v", changedWitness)
	}
	if err := ledger.record(t.Context(), target, false); err != nil {
		t.Fatal(err)
	}
	failed := prepare("complete")
	if failed.Obligation != changedWitness.Obligation || !failed.Receipt.Valid() || failed.Passed {
		t.Fatalf("failed prior receipt misreported: %+v", failed)
	}
	if err := ledger.record(t.Context(), target, true); err != nil {
		t.Fatal(err)
	}
	g.paths = []string{changed}
	g.testPlan = &testGroups{ledger: ledger, directInputs: inputs}
	g.retainSelectionCauses([]packageExecutionBatch{{Step: testOwnersCheckName, Requested: packages,
		Executions: []testevidence.PackageExecution{{Package: target, Action: "skip", Started: true}}}})
	if len(g.selectionCauses) != 1 || g.selectionCauses[0].Reuse != failed ||
		g.selectionCauses[0].Input != inputs[target] || !slices.Equal(g.selectionCauses[0].RuntimeInputs, g.paths) {
		t.Fatalf("source, runtime binding and prior receipt not joined: %+v", g.selectionCauses)
	}
	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "selection witness fixture")
	record, err := runrecord.NewGateRecord(recipe, g.environment.ID, strings.Repeat("a", 40), runrecord.OutcomeFailed, testOwnersCheckName, uint64(time.Second),
		[]runrecord.GateStep{{Name: testOwnersCheckName, Phase: runrecord.PhaseTest, Outcome: runrecord.StepFailed, DurationNS: uint64(time.Second)}})
	if err != nil {
		t.Fatal(err)
	}
	var batch artifact.Batch
	if err := g.appendSelectionCauses(&batch, record.Result.ID); err != nil {
		t.Fatal(err)
	}
	retained, err := runrecord.SelectionCauseCodec.Parse(batch.Contents[0].Data)
	if err != nil {
		t.Fatal(err)
	}
	histogram, err := runrecord.SelectionCauseHistogram(record.Result, retained)
	if err != nil || len(histogram.Packages) != 1 || histogram.Packages[0].Reuse != failed || histogram.Packages[0].Executed {
		t.Fatalf("retention lost pre-execution authority or invented execution: %+v, %v", histogram, err)
	}
	if !histogram.Packages[0].Skipped || histogram.Steps[0].Skipped != 1 || histogram.Steps[0].Unstarted != 0 ||
		histogram.Primary[0].Skipped != 1 || histogram.Primary[0].Unstarted != 0 {
		t.Fatalf("reported skip confused with unstarted work: %+v", histogram)
	}
	retained.Packages[0].Reuse.Receipt = artifact.ID{}
	retained.Packages[0].Reuse.Passed = true
	if _, err := runrecord.SelectionCauseCodec.NewInitial(retained); err == nil {
		t.Fatal("prior pass without a receipt accepted")
	}
	if short := prepare("short"); short.Obligation == failed.Obligation || short.Receipt.Valid() {
		t.Fatalf("mode change reused a complete-mode receipt: %+v", short)
	}
	ledger.environment = testutil.ArtifactID(t, artifact.KindEvidence, "different environment")
	if environment := prepare("complete"); environment.Obligation == failed.Obligation || environment.Receipt.Valid() {
		t.Fatalf("environment change reused old authority: %+v", environment)
	}
}
