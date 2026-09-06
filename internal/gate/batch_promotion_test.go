package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const promotionStorePath = "tmp/store"

// promotionRun: the fixture batch planned and executed the way the gate does
// it over an existing context; the parent acceptance check decides the
// flush; returns that check's error.
func promotionRun(t *testing.T, g *gateContext, batch *plan.VerificationBatch, tree string) error {
	t.Helper()
	g.environment = lifecycleTestEnvironment(t)
	invocations := persistenceInvocations(t, g, batch, tree)
	cache := g.loadRetryCache()
	results, err := g.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range results {
		if result.Invocation.Check.Name == "acceptance" {
			return result.Err
		}
	}
	t.Fatal("parent acceptance did not run")
	return nil
}

// commitPromotionRecord: the final record's obligation and promotion
// contents committed to the gate store with a registered result identity.
func commitPromotionRecord(t *testing.T, g *gateContext, outcome runrecord.Outcome, result artifact.ID) {
	t.Helper()
	store, err := overgodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch := artifact.Batch{Key: "gate/promotion/" + result.String(), Artifacts: []artifact.Descriptor{{ID: result}}}
	if err := g.appendCheckpointObligations(&batch, result, outcome); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
}

func outstandingObligations(t *testing.T, g *gateContext, key string) []runrecord.CheckpointObligation {
	t.Helper()
	store, err := overgodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	outstanding, err := runrecord.LoadOutstandingCheckpointObligations(t.Context(), store, key)
	if err != nil {
		t.Fatal(err)
	}
	return outstanding
}

// TestBatchPromotionControl pins: accepted checkpoints under a declared flush
// become durable obligations whether or not the run completes; an unflushed
// batch refuses parent completion; obligations survive a restart and seed the
// next candidate's flush decision; a candidate whose memo inputs differ finds
// them stale and re-accepts instead of promoting; the flush promotes every
// outstanding obligation once, bound to the full gate result.
func TestBatchPromotionControl(t *testing.T) {
	g, batch, tree := verificationBatchFixture(t, "pass")
	flush := &plan.BatchFlush{Key: "promote", MaxSize: 3, MaxInterval: "1h", MaxBytes: 1 << 20}
	batch.Flush = flush
	g.storePath = promotionStorePath
	if err := os.MkdirAll(filepath.Join(g.repo, filepath.FromSlash(promotionStorePath)), 0o755); err != nil {
		t.Fatal(err)
	}

	parentErr := promotionRun(t, g, batch, tree)
	if g.promotion == nil || g.promotion.flushed || len(g.promotion.pending) != 2 || len(g.promotion.promoted) != 2 {
		t.Fatalf("first run promotion = %+v", g.promotion)
	}
	if parentErr == nil || !strings.Contains(parentErr.Error(), "not flushed") || g.acceptedTree != "" {
		t.Fatalf("unflushed batch admitted parent completion: %v accepted=%q", parentErr, g.acceptedTree)
	}
	commitPromotionRecord(t, g, runrecord.OutcomeFailed, testutil.ArtifactID(t, artifact.KindEvidence, "refused gate result"))
	if outstanding := outstandingObligations(t, g, flush.Key); len(outstanding) != 2 {
		t.Fatalf("outstanding after refused run = %d, want both accepted checkpoints persisted", len(outstanding))
	}

	// Stale candidate: the package input differs from the obligations' input.
	unit := filepath.Join(g.repo, "unit_test.go")
	original, err := os.ReadFile(unit)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(unit, append(original, []byte("\n// touched\n")...), 0o644); err != nil {
		t.Fatal(err)
	}
	stale, staleBatch, staleTree := verificationBatchContext(t, g.repo)
	staleBatch.Flush = flush
	stale.storePath = promotionStorePath
	if err := promotionRun(t, stale, staleBatch, staleTree); err == nil {
		t.Fatal("stale candidate completed its parent without a flush")
	}
	if stale.promotion.flushed || len(stale.promotion.pending) != 2 || strings.Count(strings.Join(stale.audit, "\n"), "checkpoint obligation stale") != 2 {
		t.Fatalf("stale candidate promotion = %+v audit=%v", stale.promotion, stale.audit)
	}
	if err := os.WriteFile(unit, original, 0o644); err != nil {
		t.Fatal(err)
	}

	// Restart: the outstanding obligations seed the flush of a third checkpoint.
	restarted, restartedBatch, restartedTree := verificationBatchContext(t, g.repo)
	restartedBatch.Flush = flush
	restartedBatch.Checkpoints = append(restartedBatch.Checkpoints, plan.VerificationCheckpoint{
		ID: "integration", Title: "Integration", Verify: "go test . -run '^TestIntegration$' -count=1 -v", DependsOn: []string{"consumer"},
	})
	restarted.storePath = promotionStorePath
	if err := promotionRun(t, restarted, restartedBatch, restartedTree); err != nil || restarted.acceptedTree != restartedTree {
		t.Fatalf("flushed batch did not complete its parent: %v accepted=%q", err, restarted.acceptedTree)
	}
	decision := restarted.promotion
	if !decision.flushed || decision.reason != plan.FlushSize || len(decision.promoted) != 3 || len(decision.pending) != 1 || decision.pending[0].Checkpoint != "integration" {
		t.Fatalf("restarted promotion = %+v audit=%v", decision, restarted.audit)
	}
	result := testutil.ArtifactID(t, artifact.KindEvidence, "promoting gate result")
	commitPromotionRecord(t, restarted, runrecord.OutcomeSucceeded, result)
	if outstanding := outstandingObligations(t, restarted, flush.Key); len(outstanding) != 0 {
		t.Fatalf("outstanding after flush = %d, want every obligation promoted", len(outstanding))
	}
	store, err := overgodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	promotions, err := store.Query(t.Context(), overgodb.Query{
		Kind: artifact.KindEvidence, MediaType: runrecord.CheckpointPromotionMediaType, Schema: runrecord.CheckpointPromotionSchema,
		MaxResults: 8, Projection: overgodb.ProjectContentPresence,
	})
	if err != nil || len(promotions.Contents) != 1 {
		t.Fatalf("promotions = %d, %v; want exactly one", len(promotions.Contents), err)
	}
	promotion, err := runrecord.RequireCheckpointPromotion(t.Context(), store, promotions.Contents[0].Artifact)
	if err != nil || promotion.Result != result || len(promotion.Obligations) != 3 {
		t.Fatalf("promotion = %+v, %v", promotion, err)
	}
}
