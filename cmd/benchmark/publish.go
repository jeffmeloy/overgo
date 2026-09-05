package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// publishBenchmarkEvidence commits the measured result as evidence with
// a capability-measured verification claim, so a benchmark run is a
// store fact under the model's identity instead of terminal output
// that scrolls away. The claim carries the clean verifying commit; a
// dirty worktree refuses, the same discipline every claim path holds.
func publishBenchmarkEvidence(ctx context.Context, options options, result benchmarkResult) error {
	commit, err := runrecord.VerifyingCommit(".")
	if err != nil {
		return err
	}
	weights, err := os.Open(result.ModelPath)
	if err != nil {
		return err
	}
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	closeErr := weights.Close()
	if err != nil || closeErr != nil {
		return errors.Join(err, closeErr)
	}
	claim, evidenceData, evidence, err := benchmarkClaim(result, commit)
	if err != nil {
		return err
	}
	record, err := runrecord.NewModelVerification(model, result.ModelName, []runrecord.CapabilityClaim{claim})
	if err != nil {
		return err
	}
	store, err := overgodb.Open(options.Repository)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	if err := runrecord.CommitVerificationClaim(ctx, store, record, evidenceData, evidence, result.ModelPath); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "benchmark evidence committed: record=%s model=%s evidence=%s commit=%.12s\n",
		record.ID, model, evidence, commit)
	return nil
}

// benchmarkClaim shapes the result into one capability claim over its
// own committed bytes: measured wall is the sum of run totals, span
// tokens count generated output, and the peak device footprint rides
// along so the claim answers capacity questions without the document.
func benchmarkClaim(result benchmarkResult, commit string) (runrecord.CapabilityClaim, []byte, artifact.ID, error) {
	if err := validateBenchmarkResult(result); err != nil {
		return runrecord.CapabilityClaim{}, nil, artifact.ID{}, err
	}
	evidenceData, err := json.Marshal(result)
	if err != nil {
		return runrecord.CapabilityClaim{}, nil, artifact.ID{}, err
	}
	evidence, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(evidenceData))
	if err != nil {
		return runrecord.CapabilityClaim{}, nil, artifact.ID{}, err
	}
	var wall time.Duration
	var promptTokens uint64
	for _, run := range result.Runs {
		wall += time.Duration(run.TotalMilliseconds * float64(time.Millisecond))
		promptTokens += uint64(run.PromptTokens)
	}
	// Span fields stay zero: they declare a training span and demand a
	// dataset identity; a perf run's token counts live in the evidence.
	return runrecord.CapabilityClaim{
		Capability: "benchmark", Tier: runrecord.TierCapabilityMeasured, Commit: commit,
		ContextTokens: promptTokens,
		WallNS:        uint64(wall.Nanoseconds()), PeakDeviceBytes: result.DevicePeakBytes,
		Evidence: []artifact.ID{evidence},
	}, evidenceData, evidence, nil
}
