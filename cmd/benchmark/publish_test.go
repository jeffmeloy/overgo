package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
)

// TestBenchmarkClaimCommits pins the publish arc: the measured result
// becomes one capability-measured claim over its own committed bytes,
// the record lands in the store with the evidence document and the
// model's identity, and the committed record parses back with the
// claim intact.
func TestBenchmarkClaimCommits(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	result := benchmarkResult{
		ModelName: "fixture-model", DevicePeakBytes: 512,
		Residency: recipe.ResidencyDeviceNative, RealizedResidency: recipe.RealizedDeviceNative,
		EndOfSequenceIgnored: true, SamplingProtocol: greedyBudgetProtocol,
		TokensPerSequence: 32, RequestedRuns: 2,
		Prompt: "fixture prompt", PromptSuite: []string{"fixture prompt"},
		WorkloadDigest: benchmarkWorkloadDigest([]string{"fixture prompt"}),
		Runs: []runMetrics{
			{Run: 0, PromptDigest: benchmarkPromptDigest("fixture prompt"), OutputDigests: []string{fixtureOutputDigest(32)}, PromptTokens: 7, OutputTokens: 32, DeviceSelectedTokens: 32, TotalMilliseconds: 100},
			{Run: 1, PromptDigest: benchmarkPromptDigest("fixture prompt"), OutputDigests: []string{fixtureOutputDigest(32)}, PromptTokens: 7, OutputTokens: 32, DeviceSelectedTokens: 32, TotalMilliseconds: 120},
		},
	}
	commit := "0123456789abcdef0123456789abcdef01234567"
	bindFixtureBenchmarkIdentity(t, &result, commit)
	weightsPath := filepath.Join(t.TempDir(), "weights.gguf")
	testcheck.NoError(t, os.WriteFile(weightsPath, []byte("weights fixture bytes"), 0o644))
	weights := testcheck.Must(os.Open(weightsPath))(t)
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	weights.Close()
	testcheck.NoError(t, err)
	claim, evidenceData, evidence, err := benchmarkClaim(result, commit, model)
	testcheck.NoError(t, err)
	if claim.Capability != "benchmark" || claim.Tier != runrecord.TierCapabilityMeasured ||
		claim.Commit != commit || claim.SpanTokens != 0 || claim.ContextTokens != 14 ||
		claim.WallNS != 220_000_000 || claim.PeakDeviceBytes != 512 ||
		len(claim.Evidence) != 1 || claim.Evidence[0] != evidence {
		t.Fatalf("claim = %+v", claim)
	}
	if _, _, _, err := benchmarkClaim(result, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", model); err == nil {
		t.Fatal("a different publishing commit relabeled the measured producer")
	}
	changed := testcheck.Must(artifact.IdentifyBytes(artifact.KindModel, []byte("different weights after measurement")))(t)
	if _, _, _, err := benchmarkClaim(result, commit, changed); err == nil {
		t.Fatal("publication attached measured evidence to different weights")
	}
	record := testcheck.Must(runrecord.NewModelVerification(model, result.ModelName, []runrecord.CapabilityClaim{claim}))(t)
	store := overgodbtest.Open(t, t.TempDir())
	testcheck.NoError(t, runrecord.CommitVerificationClaim(ctx, store, record, evidenceData, evidence, weightsPath))

	content, found, err := artifact.ReadContent(ctx, store, record.ID)
	if err != nil || !found {
		t.Fatalf("record content = (%t, %v)", found, err)
	}
	parsed := testcheck.Must(runrecord.ParseModelVerification(content.Data))(t)
	if parsed.Model != model || len(parsed.Claims) != 1 || parsed.Claims[0].Evidence[0] != evidence {
		t.Fatalf("parsed record = %+v", parsed)
	}
	published, found, err := artifact.ReadContent(ctx, store, evidence)
	if err != nil || !found {
		t.Fatalf("evidence content = (%t, %v)", found, err)
	}
	var measured benchmarkResult
	testcheck.NoError(t, json.Unmarshal(published.Data, &measured))
	testcheck.NoError(t, validateBenchmarkResult(measured))
	if measured.ModelID != parsed.Model || measured.SamplingProtocol != greedyBudgetProtocol || measured.Runs[0].DeviceSelectedTokens != result.TokensPerSequence {
		t.Fatalf("publication lost sampling or observed selection path: %+v", measured)
	}
	if summary := evaluation.LatestEvidence(ctx, store, nil, 0).BenchmarksByLocation[weightsPath]; !summary.Comparison.Valid() || summary.Record != record.ID {
		t.Fatalf("published execution contract was lost by evidence reader: %+v", summary)
	}
	locations, err := store.Locations(ctx, model)
	if err != nil || len(locations) == 0 {
		t.Fatalf("model locations = (%+v, %v), want the registered weights file", locations, err)
	}
}

func fixtureOutputDigest(count int) string {
	digest := newTokenDigest()
	for range count {
		digest.add(7)
	}
	return digest.sum()
}

func bindFixtureBenchmarkIdentity(t *testing.T, result *benchmarkResult, commit string) {
	t.Helper()
	result.ModelID = testcheck.Must(artifact.IdentifyBytes(artifact.KindModel, []byte("weights fixture bytes")))(t)
	result.RecipeID = testcheck.Must(artifact.IdentifyBytes(artifact.KindRecipe, []byte("fixture recipe")))(t)
	result.TokenizerContainer = result.ModelID
	result.Device.UUID = "fixture-device"
	result.Environment = testcheck.Must(runrecord.CurrentEnvironment(result.Device.UUID, "cuda"))(t)
	result.EnvironmentID = result.Environment.ID
	result.CodeCommit = commit
	result.ModuleDigest = benchmarkPromptDigest("fixture module")
	// The adapters a run with no LoRA files holds, as the command builds them.
	result.AdapterIDs = testcheck.Must(benchmarkAdapterIDs(nil))(t)
	result.OptionsDigest = testcheck.Must(benchmarkOptionsDigest(options{Tokens: result.TokensPerSequence,
		Runs: result.RequestedRuns, Warmup: result.RequestedWarmup, CachePrompt: result.CachePrompt,
		BatchSequences: result.BatchSequences, ContextShift: result.ContextShift, Speculative: result.Speculative,
		Temperature: result.Temperature, TopK: result.TopK, DeviceTopK: result.DeviceTopK}, result.AdapterIDs))(t)
}
