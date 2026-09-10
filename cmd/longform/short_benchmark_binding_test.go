package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Resolve the embedded calibration, never the model's latest benchmark.
func checkGuardShortBenchmark(ctx context.Context, store *overgodb.Store, result longform.Result) error {
	id := result.ShortBenchmark
	if !id.Valid() || id.Kind() != artifact.KindEvidence {
		return fmt.Errorf("guard calibration: invalid benchmark record %s", id)
	}
	content, found, err := artifact.ReadContent(ctx, store, id)
	if err != nil || !found {
		return fmt.Errorf("guard calibration: record %s absent: %v", id, err)
	}
	record, err := runrecord.ParseModelVerification(content.Data)
	if err != nil || record.ID != id || record.Model != result.Inputs.Model || len(record.Claims) != 1 {
		return fmt.Errorf("guard calibration: record identity or weights differ: %v", err)
	}
	claim := record.Claims[0]
	if claim.Capability != "benchmark" || claim.Tier != runrecord.TierCapabilityMeasured || len(claim.Evidence) != 1 {
		return errors.New("guard calibration: expected one measured benchmark claim and body")
	}
	content, found, err = artifact.ReadContent(ctx, store, claim.Evidence[0])
	if err != nil || !found {
		return fmt.Errorf("guard calibration: benchmark body absent: %v", err)
	}
	body, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(content.Data))
	if err != nil || body != claim.Evidence[0] {
		return fmt.Errorf("guard calibration: benchmark body identity differs: %v", err)
	}
	var measured struct {
		Summary evaluation.BenchmarkSummary `json:"summary"`
	}
	if err := json.Unmarshal(content.Data, &measured); err != nil {
		return fmt.Errorf("guard calibration: benchmark body: %w", err)
	}
	rates := longform.ShortRates{
		PromptTokensPerSecond: measured.Summary.PromptTokensPerSecond,
		DecodeTokensPerSecond: measured.Summary.DecodeTokensPerSecond,
	}
	if !validShortRates(rates) || rates != result.Short {
		return fmt.Errorf("guard calibration: embedded rates differ from benchmark %s", id)
	}
	return nil
}

func TestGuardShortBenchmarkBinding(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	result := guardResult(t)
	data, err := json.Marshal(struct {
		Summary evaluation.BenchmarkSummary `json:"summary"`
	}{evaluation.BenchmarkSummary{
		PromptTokensPerSecond: result.Short.PromptTokensPerSecond,
		DecodeTokensPerSecond: result.Short.DecodeTokensPerSecond,
	}})
	if err != nil {
		t.Fatal(err)
	}
	body, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	publish := func(capability string, tier runrecord.VerificationTier) artifact.ID {
		t.Helper()
		record, err := runrecord.NewModelVerification(result.Inputs.Model, "calibration fixture", []runrecord.CapabilityClaim{{
			Capability: capability, Tier: tier, Commit: result.Commit, Evidence: []artifact.ID{body},
		}})
		if err != nil {
			t.Fatal(err)
		}
		if err := runrecord.CommitVerificationClaim(t.Context(), store, record, data, body, result.ModelPath); err != nil {
			t.Fatal(err)
		}
		return record.ID
	}
	result.ShortBenchmark = publish("benchmark", runrecord.TierCapabilityMeasured)
	if err := checkGuardShortBenchmark(t.Context(), store, result); err != nil {
		t.Fatal(err)
	}
	wrongCapability := publish("long-form", runrecord.TierCapabilityMeasured)
	wrongTier := publish("benchmark", runrecord.TierRealArtifactSmoke)
	otherModel, _, err := artifact.Identify(artifact.KindModel, bytes.NewReader([]byte("other weights")))
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*longform.Result){
		"missing record":      func(r *longform.Result) { r.ShortBenchmark = artifact.ID{} },
		"wrong kind":          func(r *longform.Result) { r.ShortBenchmark = r.Inputs.Model },
		"body used as record": func(r *longform.Result) { r.ShortBenchmark = body },
		"wrong weights":       func(r *longform.Result) { r.Inputs.Model = otherModel },
		"wrong capability":    func(r *longform.Result) { r.ShortBenchmark = wrongCapability },
		"unmeasured claim":    func(r *longform.Result) { r.ShortBenchmark = wrongTier },
		"changed prompt rate": func(r *longform.Result) { r.Short.PromptTokensPerSecond *= 2 },
		"changed decode rate": func(r *longform.Result) { r.Short.DecodeTokensPerSecond *= 2 },
	} {
		t.Run(name, func(t *testing.T) {
			changed := result
			mutate(&changed)
			if err := checkGuardShortBenchmark(t.Context(), store, changed); err == nil {
				t.Fatal("invalid calibration accepted")
			}
		})
	}
	if err := checkGuardShortBenchmark(t.Context(), store, result); err != nil {
		t.Fatalf("later claims displaced the exact historical calibration: %v", err)
	}
}
