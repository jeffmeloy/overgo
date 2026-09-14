package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestCheckpointCacheMeasurementEligibility(t *testing.T) {
	t.Parallel()
	input := testutil.ArtifactID(t, artifact.KindProfile, "phase input")
	memo := checkpointMemoEntry{
		slot:  testutil.ArtifactID(t, artifact.KindRecipe, "memo slot"),
		input: testutil.ArtifactID(t, artifact.KindProfile, "memo input"),
	}
	g := gateContext{checkpointMemos: map[string]checkpointMemoEntry{"acceptance-reference": memo}}
	for _, test := range []struct {
		name     string
		input    bool
		eligible bool
	}{
		{"acceptance-reference", true, true},
		{"acceptance-reference", false, true},
		{"vet", true, true},
		{"vet", false, false},
		{"test", true, false},
		{"test", false, false},
		{"acceptance", true, false},
		{"acceptance-undeclared", true, false},
		{"device", true, false},
		{"commit", true, false},
	} {
		check := automationcheck.Invocation{
			ID:    testutil.ArtifactID(t, artifact.KindRecipe, test.name),
			Check: automationcheck.Descriptor{Name: test.name},
		}
		var phaseInput artifact.ID
		if test.input {
			phaseInput = input
		}
		slot, actualInput, eligible := g.checkCacheKey(check, phaseInput)
		if eligible != test.eligible {
			t.Fatalf("%s input=%t eligible=%t, want %t", test.name, test.input, eligible, test.eligible)
		}
		if test.name == "acceptance-reference" {
			if slot.ID != memo.slot || actualInput != memo.input {
				t.Fatal("checkpoint measurement did not select the execution memo")
			}
		} else if slot.ID != check.ID || test.input && actualInput != input {
			t.Fatal("ordinary phase cache identity changed")
		}
	}

	// A reused checkpoint must be eligible even when no ordinary phase ran.
	// The old accounting produced hits=1, eligible=0 and could not publish the
	// manifest analysis of a failed gate. Exercise the immutable record codec.
	g2, batch, tree := verificationBatchFixture(t, "pass")
	checks, err := g2.batchAcceptanceChecks([]automationcheck.Check{
		gateCheck("acceptance", runrecord.PhaseTest, g2.stepAcceptance),
	}, batch)
	if err != nil {
		t.Fatal(err)
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := automationcheck.BindManifestPlan(
		testutil.ArtifactID(t, artifact.KindProfile, "measurement base"),
		testutil.ArtifactID(t, artifact.KindProfile, "measurement candidate"),
		strings.Repeat("a", 64), candidateTreeKey(tree),
		automationcheck.Surface{Identity: "checkpoint measurement fixture"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	eligible := 0
	if _, _, canReuse := g2.checkCacheKey(invocations[0], artifact.ID{}); canReuse {
		eligible++
	}
	measurements := automationcheck.MeasureManifest(len(checks), len(invocations), 0, 0, eligible, 1, 0, 0)
	analysis, err := automationcheck.NewManifestAnalysis(
		codemanifest.Delta{Base: manifest.BaseManifest, Candidate: manifest.CandidateManifest},
		codemanifest.Impact{Base: manifest.BaseManifest.String(), Candidate: manifest.CandidateManifest.String()},
		manifest, automationcheck.SelectionMetrics{}, measurements,
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := analysis.Content()
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := automationcheck.ParseManifestAnalysis(content.Data)
	if err != nil || replayed.ID != analysis.ID || replayed.Measurements.CacheMisses != 0 {
		t.Fatalf("checkpoint measurement round trip: %+v, %v", replayed.Measurements, err)
	}
}

// TestCheckpointEvidenceReuseAcrossRuns pins: memo slots per checkpoint; an
// executed checkpoint is reused on the same input with Reused set; a changed
// key or changed package source misses; a verify without a package is refused
// with an audited reason and no memo.
func TestCheckpointEvidenceReuseAcrossRuns(t *testing.T) {
	t.Parallel()
	g, batch, tree := verificationBatchFixture(t, "pass")
	batch.Flush = &plan.BatchFlush{Key: "fixture", MaxSize: 2, MaxInterval: "1m", MaxBytes: 1 << 20}
	checks, err := g.batchAcceptanceChecks([]automationcheck.Check{
		gateCheck("acceptance", runrecord.PhaseTest, g.stepAcceptance),
	}, batch)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.checkpointMemos) != 2 {
		t.Fatalf("memo entries = %d, want one per checkpoint", len(g.checkpointMemos))
	}
	producerMemo, consumerMemo := g.checkpointMemos["acceptance-producer"], g.checkpointMemos["acceptance-consumer"]
	if producerMemo.slot == consumerMemo.slot || producerMemo.input == consumerMemo.input {
		t.Fatal("checkpoints share a memo slot or input")
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	// Bind the same immutable candidate the acceptance fixture binds, so the
	// checkpoint executes against the manifest plan it was planned under.
	manifest, err := automationcheck.BindManifestPlan(
		testutil.ArtifactID(t, artifact.KindProfile, "memo base"),
		testutil.ArtifactID(t, artifact.KindProfile, "memo candidate"),
		strings.Repeat("a", 64), candidateTreeKey(tree),
		automationcheck.Surface{Identity: "checkpoint memo fixture"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	g.manifestPlan = &manifest
	var producer automationcheck.Invocation
	for index, invocation := range invocations {
		invocations[index], err = g.bindCheckExecution(manifest, invocation, manifest.CandidateManifest)
		if err != nil {
			t.Fatal(err)
		}
		if invocation.Check.Name == "acceptance-producer" {
			producer = invocations[index]
		}
	}
	slot, input, memoised := g.memoSlot(producer)
	if !memoised || slot.ID != producerMemo.slot || input != producerMemo.input || slot.Check.Name != "acceptance-producer" {
		t.Fatalf("memo slot = %+v %s %v", slot, input, memoised)
	}
	if _, _, memoised := g.memoSlot(invocations[len(invocations)-1]); memoised {
		t.Fatal("the parent acceptance check must not be memoised")
	}

	cache := automationcheck.NewEvidenceCache(g.environment.ID)
	if _, found := cache.Lookup(slot, input); found {
		t.Fatal("empty cache reused a checkpoint")
	}
	evidence, err := automationcheck.Run(t.Context(), producer)
	if err != nil || evidence.Outcome != runrecord.LanePassed {
		t.Fatalf("producer checkpoint = %+v, %v", evidence, err)
	}
	cache.Record(slot, input, evidence)
	reused, found := cache.Lookup(slot, input)
	if !found || !reused.Reused || reused.Name != "acceptance-producer" {
		t.Fatalf("same input did not reuse: %+v %v", reused, found)
	}

	again, err := g.checkpointMemoInputs(batch)
	if err != nil || again["acceptance-producer"] != producerMemo {
		t.Fatalf("recomputed memo differs on an unchanged tree: %+v %v", again["acceptance-producer"], err)
	}
	// This root-package fixture reads arbitrary files. Documentation changes
	// remain inputs until a narrower read contract proves independence.
	if err := os.WriteFile(filepath.Join(g.repo, "NOTES.md"), []byte("Corrected documentation.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g.packageGraph = nil
	repaired, err := g.checkpointMemoInputs(batch)
	if err != nil || repaired["acceptance-producer"].input == producerMemo.input {
		t.Fatalf("opaque file reader retained stale checkpoint input: %v", err)
	}
	if _, found := cache.Lookup(slot, repaired["acceptance-producer"].input); found {
		t.Fatal("opaque file reader reused stale checkpoint evidence")
	}
	if err := os.Remove(filepath.Join(g.repo, "NOTES.md")); err != nil {
		t.Fatal(err)
	}
	g.packageGraph = nil
	restored, err := g.checkpointMemoInputs(batch)
	if err != nil || restored["acceptance-producer"] != producerMemo {
		t.Fatalf("restored package inputs lost their original memo: %v", err)
	}
	batch.Flush.Key = "other"
	other, err := g.checkpointMemoInputs(batch)
	if err != nil || other["acceptance-producer"].slot == producerMemo.slot || other["acceptance-producer"].input == producerMemo.input {
		t.Fatalf("changed key kept slot or input: %v", err)
	}
	batch.Flush.Key = "fixture"

	source := filepath.Join(g.repo, "unit_test.go")
	data, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, append(data, []byte("\n// changed\n")...), 0o644); err != nil {
		t.Fatal(err)
	}
	g.packageGraph = nil
	changed, err := g.checkpointMemoInputs(batch)
	if err != nil || changed["acceptance-producer"].slot != producerMemo.slot || changed["acceptance-producer"].input == producerMemo.input {
		t.Fatalf("changed source kept the memo input or moved the slot: %v", err)
	}
	if _, found := cache.Lookup(slot, changed["acceptance-producer"].input); found {
		t.Fatal("changed source reused stale checkpoint evidence")
	}

	bad := *batch
	bad.Checkpoints = []plan.VerificationCheckpoint{{ID: "bare", Title: "Bare", Verify: "go test -run '^TestProducer$' -count=1"}}
	g.audit = nil
	memos, err := g.checkpointMemoInputs(&bad)
	if err != nil || len(memos) != 0 || len(g.audit) != 1 || !strings.Contains(g.audit[0], "bare: go test segment names no package") {
		t.Fatalf("verify without a package: memos=%v err=%v audit=%v; want an audited refusal", memos, err, g.audit)
	}
}
