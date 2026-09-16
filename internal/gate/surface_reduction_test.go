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
	"overgo/internal/codemanifest"
	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestSurfacePreflightComposition(t *testing.T) {
	t.Parallel()
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	for _, retired := range []string{"cmd/preflight/main.go", "internal/model/family_branch_census.go"} {
		if _, err := os.Stat(filepath.Join(root, retired)); !os.IsNotExist(err) {
			t.Fatalf("retired entry remains: %s %v", retired, err)
		}
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	baselinePath := filepath.Join(root, harnessSurfaceBaselineFile)
	before, err := os.ReadFile(baselinePath)
	if err != nil {
		t.Fatal(err)
	}
	g := &gateContext{repo: root, source: &snapshot, preflight: true}
	if err := g.architectureDiagnostics(snapshot); err != nil {
		t.Fatal(err)
	}
	for _, fixture := range []struct{ path, source, refusal string }{
		{"internal/shared/fixture.go", "package shared\nfunc Forbidden(s string) bool { return s == \"qwen999\" }", "family-named branch"},
		{"internal/automationpolicy/fixture.go", "package automationpolicy\nfunc Legacy() {}", "harness consolidation"},
	} {
		changed, err := snapshot.Overlay(map[string][]byte{fixture.path: []byte(fixture.source)})
		if err != nil {
			t.Fatal(err)
		}
		g.source = &changed
		if err := g.architectureDiagnostics(changed); err == nil || !strings.Contains(err.Error(), fixture.refusal) {
			t.Fatalf("lost %s refusal: %v", fixture.refusal, err)
		}
	}
	after, err := os.ReadFile(baselinePath)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("diagnostic changed baseline: %v", err)
	}
	fixtureRoot := t.TempDir()
	runGitFixture(t, fixtureRoot, "init")
	const generated = "docs/api_manifest.json"
	testutil.WriteTextFile(t, fixtureRoot, generated, "{}\n")
	scope := &gateContext{repo: fixtureRoot, paths: []string{generated}, preflight: true}
	if _, err := scope.stepScope(); err != nil || scope.pendingGenerated == nil || *scope.pendingGenerated != 1 {
		t.Fatalf("lost generated-file advisory: %v %+v", err, scope.pendingGenerated)
	}
	testutil.WriteTextFile(t, fixtureRoot, "nested/plan.json", "{}\n")
	if _, err := scope.stepScope(); err == nil || !strings.Contains(err.Error(), "competing live plan") {
		t.Fatalf("nested plan admitted: %v", err)
	}
}

func TestSurfaceProfileReuse(t *testing.T) {
	root := t.TempDir()
	const name = "internal/example/example.go"
	testutil.WriteTextFile(t, root, name, "package example\nfunc Value() int { return 1 }\n")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	want, err := codeprofile.Build(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	g := &gateContext{}
	if got, err := g.sourceProfile(snapshot); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("uncached profile: %v", err)
	}
	g.manifestCache, err = codemanifest.NewCache(1)
	if err != nil {
		t.Fatal(err)
	}
	selection := repoanalysis.BuildSelection{Context: "linux/amd64", Root: root, Files: map[string]bool{name: true}, Packages: map[string]string{name: "overgo/internal/example"}}
	if _, _, err := g.manifestCache.Generate(snapshot, []repoanalysis.BuildSelection{selection}, nil); err != nil {
		t.Fatal(err)
	}
	for _, label := range []string{"first reader", "second reader"} {
		t.Run(label, func(t *testing.T) {
			t.Parallel()
			got, err := g.sourceProfile(snapshot)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("shared profile: %v", err)
			}
			got.Functions[0].Name = label
			again, err := g.sourceProfile(snapshot)
			if err != nil || !reflect.DeepEqual(again, want) {
				t.Fatalf("caller mutation reached cache: %v", err)
			}
		})
	}
}

func TestSurfaceSelectionWitness(t *testing.T) {
	t.Parallel()
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
	testutil.WriteTextFile(t, g.repo, changed, "2\n")
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
	if notes := compactAudit(g.audit); len(notes) != 1 || !strings.Contains(notes[0], batch.Contents[0].Descriptor.ID.String()) || !strings.Contains(notes[0], "plan -history") {
		t.Fatalf("retained selection query is not discoverable: %v", notes)
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
