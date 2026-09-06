package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestCheckpointEvidenceReuseAcrossRuns pins: memo slots per checkpoint; an
// executed checkpoint is reused on the same input with Reused set; a changed
// key or changed package source misses; a verify without a package is refused.
func TestCheckpointEvidenceReuseAcrossRuns(t *testing.T) {
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
		invocations[index], err = automationcheck.BindManifestExecution(manifest, invocation, []artifact.ID{manifest.CandidateManifest})
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

	cache := automationcheck.NewEvidenceCache(testutil.ArtifactID(t, artifact.KindProfile, "memo environment"))
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
	if _, err := g.checkpointMemoInputs(&bad); err == nil {
		t.Fatal("verify without a package produced a memo")
	}
}
