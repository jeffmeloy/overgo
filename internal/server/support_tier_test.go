package server

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestModelSupportTier: a committed long-form record joins its model's
// weights location with its verdict, commit and surface; the catalog grades
// every row from its activations and that record -- registered alone,
// self-consistent under a verified activation, parity-checked under parity
// evidence, guarded only by a passing record, trainable under an active
// training recipe -- and a stale row stays registered with its reason.
func TestModelSupportTier(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	ctx := t.Context()
	store := handler.config.Repository
	weightsPath := filepath.Join(t.TempDir(), "guarded.gguf")
	if err := os.WriteFile(weightsPath, []byte("support tier weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	weights, err := os.Open(weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	weights.Close()
	if err != nil {
		t.Fatal(err)
	}
	evidenceData := []byte(`{"commit":"0123456789abcdef0123","surface":"feedfacecafebeef0011","verdict":{"passed":true},"output":"long text"}`)
	evidence, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(evidenceData))
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewModelVerification(model, "fixture", []runrecord.CapabilityClaim{{
		Capability: "long-form", Tier: runrecord.TierCapabilityMeasured,
		Commit: "0123456789abcdef0123456789abcdef01234567", Evidence: []artifact.ID{evidence},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := runrecord.CommitVerificationClaim(ctx, store, record, evidenceData, evidence, weightsPath); err != nil {
		t.Fatal(err)
	}
	registered, err := filepath.Abs(weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	index := handler.catalogEvidence(ctx, nil)
	guard, found := index.GuardsByLocation[registered]
	if !found || !guard.Verdict.Passed || guard.Commit != "0123456789abcdef0123" || guard.Surface != "feedfacecafebeef0011" || guard.Record != record.ID {
		t.Fatalf("guard summary = (%+v, %t) keys=%v", guard, found, index.GuardsByLocation)
	}

	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "support-inference")
	training := testutil.ArtifactID(t, artifact.KindRecipe, "support-training")
	entry := func(location string, capabilities ...discovery.Capability) discovery.CatalogEntry {
		return discovery.CatalogEntry{Model: model, Location: location, Present: true, Capabilities: capabilities}
	}
	inference := func(tier recipe.EvidenceTier, stale string) discovery.Capability {
		return discovery.Capability{Task: recipe.TaskInference, Recipe: recipeID, Tier: tier, Stale: stale}
	}
	rows := catalogListing([]discovery.CatalogEntry{
		entry(registered, inference(recipe.EvidenceVerified, ""), discovery.Capability{Task: recipe.TaskTraining, Recipe: training, Tier: recipe.EvidenceVerified}),
		entry("C:/models/parity.gguf", inference(recipe.EvidenceParity, "")),
		entry("C:/models/stale.gguf", inference(recipe.EvidenceParity, "architecture qwen9 is not supported")),
		entry("C:/models/bare.gguf"),
	}, index)
	held := func(support modelSupport) map[string]string {
		rungs := map[string]string{}
		for _, rung := range support.Rungs {
			if rung.Held {
				rungs[rung.Name] = rung.Detail
			}
		}
		return rungs
	}
	guarded := held(rows[0].Support)
	if rows[0].Support.Tier != "self-consistent" || len(guarded) != 4 || guarded["parity-checked"] != "" ||
		!strings.Contains(guarded["guarded"], "0123456789ab") || !strings.Contains(guarded["guarded"], "feedfacecafe") ||
		!strings.Contains(guarded["trainable"], training.String()) {
		t.Fatalf("guarded row support = %+v", rows[0].Support)
	}
	if parity := held(rows[1].Support); rows[1].Support.Tier != "parity-checked" || len(parity) != 3 {
		t.Fatalf("parity row support = %+v", rows[1].Support)
	}
	stale := rows[2].Support
	if stale.Tier != "registered" || len(held(stale)) != 1 || stale.Rungs[1].Detail != "unservable: architecture qwen9 is not supported" {
		t.Fatalf("stale row support = %+v", stale)
	}
	if bare := rows[3].Support; bare.Tier != "registered" || len(held(bare)) != 1 || bare.Rungs[3].Detail != "no long-form record" {
		t.Fatalf("bare row support = %+v", bare)
	}

	// A failed long-form record names its failure and holds no guard.
	failed := evaluation.GuardSummary{Commit: "abcdef0123456789"}
	if support := catalogSupport(rows[3], failed, true); support.Rungs[3].Held || support.Rungs[3].Detail != "long-form guard failed its floors at commit abcdef012345" {
		t.Fatalf("failed guard support = %+v", support)
	}
}

// TestUnreadableGuardEvidenceIsNoGuard holds a long-form record whose evidence
// cannot be read to no guard at all: the model is not shown as having failed
// its floors at a commit the record never stated.
func TestUnreadableGuardEvidenceIsNoGuard(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	ctx := t.Context()
	store := handler.config.Repository
	weightsPath := filepath.Join(t.TempDir(), "unreadable.gguf")
	if err := os.WriteFile(weightsPath, []byte("unreadable guard weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	weights, err := os.Open(weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	weights.Close()
	if err != nil {
		t.Fatal(err)
	}
	evidenceData := []byte("the long-form record is not a document")
	evidence, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(evidenceData))
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewModelVerification(model, "fixture", []runrecord.CapabilityClaim{{
		Capability: "long-form", Tier: runrecord.TierCapabilityMeasured,
		Commit: "0123456789abcdef0123456789abcdef01234567", Evidence: []artifact.ID{evidence},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := runrecord.CommitVerificationClaim(ctx, store, record, evidenceData, evidence, weightsPath); err != nil {
		t.Fatal(err)
	}
	registered, err := filepath.Abs(weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	if guard, found := handler.catalogEvidence(ctx, nil).GuardsByLocation[registered]; found {
		t.Fatalf("unreadable evidence became a guard: %+v", guard)
	}
}
