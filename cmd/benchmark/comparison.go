// Package main measures local inference and emits lossless paired benchmark evidence.
package main

import (
	"errors"
	"reflect"

	"overgo/internal/artifact"
	"overgo/internal/benchmarkrecord"
)

type pairedBenchmarkResult struct {
	Baseline     benchmarkResult                           `json:"baseline"`
	Candidate    benchmarkResult                           `json:"candidate"`
	Comparison   benchmarkrecord.BenchmarkComparisonRecord `json:"comparison"`
	ComparisonID artifact.ID                               `json:"comparison_id"`
}

func benchmarkObservation(run runMetrics) benchmarkrecord.BenchmarkObservation {
	return benchmarkrecord.BenchmarkObservation{
		PromptDigest: run.PromptDigest, TokenDigests: run.OutputDigests, OutputTokens: run.OutputTokens,
		PromptTokens: run.PromptTokens, CachedPromptTokens: run.CachedPromptTokens,
		HostLogitTokens: run.HostLogitTokens, DeviceSelectedTokens: run.DeviceSelectedTokens,
		DeviceTopKTokens: run.DeviceTopKTokens, PromptMilliseconds: run.PromptMilliseconds,
		PromptTokensPerSecond:   run.PromptTokensPerSecond,
		EndToEndTokensPerSecond: run.EndToEndTokensPerSecond,
		DecodeTokensPerSecond:   run.DecodeTokensPerSecond,
		HostToDeviceCopies:      run.HostToDeviceCopies, DeviceToHostCopies: run.DeviceToHostCopies,
		DeviceToDeviceCopies: run.DeviceToDeviceCopies, DeviceToDeviceBytes: run.DeviceToDeviceBytes,
		DeviceMemsets: run.DeviceMemsets, DeviceMemsetBytes: run.DeviceMemsetBytes,
		GraphInstantiations: run.GraphInstantiations, GraphUpdates: run.GraphUpdates,
		GraphLaunches:     run.GraphLaunches,
		TotalMilliseconds: run.TotalMilliseconds, TTFTMilliseconds: run.TTFTMilliseconds,
		DecodeMilliseconds: run.DecodeMilliseconds, KernelLaunches: run.KernelLaunches,
		StreamSynchronizations: run.StreamSynchronizations,
		HostToDeviceBytes:      run.HostToDeviceBytes, DeviceToHostBytes: run.DeviceToHostBytes,
	}
}

func benchmarkResultConfiguration(result benchmarkResult) benchmarkrecord.BenchmarkConfiguration {
	return benchmarkrecord.BenchmarkConfiguration{
		Tokens: result.TokensPerSequence, Runs: result.RequestedRuns, Warmup: result.RequestedWarmup,
		CachePrompt: result.CachePrompt, BatchSequences: result.BatchSequences,
		ContextShift: result.ContextShift, Speculative: result.Speculative,
		Temperature: result.Temperature, TopK: result.TopK, DeviceTopK: result.DeviceTopK,
		Adapters: result.AdapterIDs,
	}
}

func compareBenchmarkResults(baseline, candidate benchmarkResult, factor string, first []string) (benchmarkrecord.BenchmarkComparisonRecord, error) {
	if err := validateBenchmarkResult(baseline); err != nil {
		return benchmarkrecord.BenchmarkComparisonRecord{}, err
	}
	if err := validateBenchmarkResult(candidate); err != nil {
		return benchmarkrecord.BenchmarkComparisonRecord{}, err
	}
	if len(first) != len(baseline.Runs) || len(candidate.Runs) != len(baseline.Runs) ||
		baseline.ModelID != candidate.ModelID || baseline.RecipeID != candidate.RecipeID ||
		baseline.TokenizerContainer != candidate.TokenizerContainer ||
		baseline.EnvironmentID != candidate.EnvironmentID || baseline.CodeCommit != candidate.CodeCommit ||
		baseline.ModuleDigest != candidate.ModuleDigest || baseline.WorkloadDigest != candidate.WorkloadDigest ||
		baseline.RealizedResidency != candidate.RealizedResidency ||
		!reflect.DeepEqual(baseline.PromptSuite, candidate.PromptSuite) ||
		baseline.Device != candidate.Device || baseline.Residency != candidate.Residency {
		return benchmarkrecord.BenchmarkComparisonRecord{}, errors.New("benchmark: paired results have incompatible artifact, environment, workload or realized residency")
	}
	promptDigests := make([]string, len(baseline.PromptSuite))
	for index, prompt := range baseline.PromptSuite {
		promptDigests[index] = benchmarkPromptDigest(prompt)
	}
	warmups := make([]benchmarkrecord.BenchmarkObservation, 0, len(baseline.WarmupRuns)+len(candidate.WarmupRuns))
	for index := range baseline.WarmupRuns {
		warmups = append(warmups, benchmarkObservation(baseline.WarmupRuns[index]), benchmarkObservation(candidate.WarmupRuns[index]))
	}
	pairs := make([]benchmarkrecord.BenchmarkPair, len(baseline.Runs))
	for index := range pairs {
		pairs[index] = benchmarkrecord.BenchmarkPair{Index: index, First: first[index],
			Baseline:  benchmarkObservation(baseline.Runs[index]),
			Candidate: benchmarkObservation(candidate.Runs[index])}
	}
	return benchmarkrecord.NewBenchmarkComparisonRecord(benchmarkrecord.BenchmarkComparisonRecord{
		Identity: benchmarkrecord.BenchmarkIdentity{
			Model: baseline.ModelID, Recipe: baseline.RecipeID,
			TokenizerContainer: baseline.TokenizerContainer, Environment: baseline.EnvironmentID,
			CodeCommit: baseline.CodeCommit, ModuleDigest: baseline.ModuleDigest,
			WorkloadDigest: baseline.WorkloadDigest, PromptDigests: promptDigests,
			RealizedResidency: baseline.RealizedResidency,
		},
		Factor: factor, Baseline: benchmarkResultConfiguration(baseline),
		Candidate: benchmarkResultConfiguration(candidate), Exclusive: true,
		SharedLoadMilliseconds: baseline.LoadMilliseconds,
		SharedDevicePeakBytes:  baseline.DevicePeakBytes,
		Warmups:                warmups, Pairs: pairs,
		CacheUpperBound: factor == "cache_prompt" && len(promptDigests) == 1,
	})
}
