package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestBenchmarkClaimCommits pins the publish arc: the measured result
// becomes one capability-measured claim over its own committed bytes,
// the record lands in the store with the evidence document and the
// model's identity, and the committed record parses back with the
// claim intact.
func TestBenchmarkClaimCommits(t *testing.T) {
	ctx := t.Context()
	result := benchmarkResult{
		ModelName: "fixture-model", DevicePeakBytes: 512,
		Runs: []runMetrics{
			{Run: 0, PromptTokens: 7, OutputTokens: 32, TotalMilliseconds: 100},
			{Run: 1, PromptTokens: 7, OutputTokens: 32, TotalMilliseconds: 120},
		},
	}
	commit := "0123456789abcdef0123456789abcdef01234567"
	claim, evidenceData, evidence, err := benchmarkClaim(result, commit)
	if err != nil {
		t.Fatal(err)
	}
	if claim.Capability != "benchmark" || claim.Tier != runrecord.TierCapabilityMeasured ||
		claim.Commit != commit || claim.SpanTokens != 0 || claim.ContextTokens != 14 ||
		claim.WallNS != 220_000_000 || claim.PeakDeviceBytes != 512 ||
		len(claim.Evidence) != 1 || claim.Evidence[0] != evidence {
		t.Fatalf("claim = %+v", claim)
	}

	weightsPath := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(weightsPath, []byte("weights fixture bytes"), 0o644); err != nil {
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
	record, err := runrecord.NewModelVerification(model, result.ModelName, []runrecord.CapabilityClaim{claim})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := runrecord.CommitVerificationClaim(ctx, store, record, evidenceData, evidence, weightsPath); err != nil {
		t.Fatal(err)
	}

	content, found, err := artifact.ReadContent(ctx, store, record.ID)
	if err != nil || !found {
		t.Fatalf("record content = (%t, %v)", found, err)
	}
	parsed, err := runrecord.ParseModelVerification(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Model != model || len(parsed.Claims) != 1 || parsed.Claims[0].Evidence[0] != evidence {
		t.Fatalf("parsed record = %+v", parsed)
	}
	if _, found, err := artifact.ReadContent(ctx, store, evidence); err != nil || !found {
		t.Fatalf("evidence content = (%t, %v)", found, err)
	}
	locations, err := store.Locations(ctx, model)
	if err != nil || len(locations) == 0 {
		t.Fatalf("model locations = (%+v, %v), want the registered weights file", locations, err)
	}
}
