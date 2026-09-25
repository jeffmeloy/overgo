package main

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/sampling"
)

const (
	greedyBudgetProtocol = "greedy/continue-after-eog/v1"
	topKBudgetProtocol   = "top-k-temperature/continue-after-eog/v1"
)

func benchmarkProtocol(temperature float64) string {
	if temperature == 0 {
		return greedyBudgetProtocol
	}
	return topKBudgetProtocol
}

func benchmarkSamplingConfig(options options) sampling.Config {
	config := sampling.Config{Temperature: float32(options.Temperature), TopK: options.TopK}
	if options.Temperature > 0 {
		config.Samplers = []sampling.SamplerStage{sampling.SamplerTopK, sampling.SamplerTemperature}
	}
	return config
}

func benchmarkSampler(options options) (*sampling.Sampler, error) {
	return sampling.New(benchmarkSamplingConfig(options))
}

func benchmarkGenerationOptions(options options) (inference.GenerateOptions, error) {
	sampler, err := benchmarkSampler(options)
	if err != nil {
		return inference.GenerateOptions{}, err
	}
	return inference.GenerateOptions{MaxNewTokens: options.Tokens, Sampler: sampler,
		ContinueAfterEOG: true, DeviceGreedy: sampler.IsRawGreedy(),
		SpeculativeDecode: options.Speculative, ContextShift: options.ContextShift, CachePrompt: options.CachePrompt,
	}, nil
}

// validateBenchmarkResult refuses to publish a legacy or incomplete measurement
// as the new protocol. Stored historical documents keep their original bytes.
func validateBenchmarkResult(result benchmarkResult) error {
	if !result.ModelID.Valid() || result.ModelID.Kind() != artifact.KindModel ||
		!result.RecipeID.Valid() || result.RecipeID.Kind() != artifact.KindRecipe ||
		result.TokenizerContainer != result.ModelID || !result.EnvironmentID.Valid() ||
		result.EnvironmentID.Kind() != artifact.KindEvidence {
		return errors.New("benchmark: model, recipe, tokenizer or environment identity is unbound")
	}
	environment := result.Environment
	environment.ID = result.EnvironmentID
	if err := environment.ValidateIdentity(); err != nil {
		return err
	}
	if result.Environment.Device != result.Device.UUID || !validIdentityDigest(result.ModuleDigest) ||
		!validIdentityDigest(result.OptionsDigest) ||
		(len(result.CodeCommit) != 40 && len(result.CodeCommit) != 64) || !validHex(result.CodeCommit) {
		return errors.New("benchmark: exact device, code or module identity is unbound")
	}
	for _, adapter := range result.AdapterIDs {
		if !adapter.Valid() || adapter.Kind() != artifact.KindAdapter {
			return errors.New("benchmark: adapter identity is unbound")
		}
	}
	optionsDigest, err := benchmarkOptionsDigest(options{Tokens: result.TokensPerSequence, Runs: result.RequestedRuns,
		Warmup: result.RequestedWarmup, CachePrompt: result.CachePrompt, BatchSequences: result.BatchSequences,
		ContextShift: result.ContextShift, Speculative: result.Speculative, Temperature: result.Temperature,
		TopK: result.TopK, DeviceTopK: result.DeviceTopK}, result.AdapterIDs)
	if err != nil || result.OptionsDigest != optionsDigest {
		return errors.New("benchmark: options digest differs from measured configuration")
	}
	if result.Residency == "" || !result.Residency.Valid() || !result.RealizedResidency.Valid() {
		return errors.New("benchmark: requested or realized residency is unbound")
	}
	opts := options{Temperature: result.Temperature, TopK: result.TopK}
	if _, err := benchmarkSampler(opts); err != nil {
		return err
	}
	if !result.EndOfSequenceIgnored || result.SamplingProtocol != benchmarkProtocol(result.Temperature) ||
		!slices.Equal(result.SamplerOrder, benchmarkSamplingConfig(opts).Samplers) {
		return errors.New("benchmark: sampling protocol is absent or differs from the measured configuration")
	}
	if len(result.PromptSuite) == 0 || result.Prompt != result.PromptSuite[0] ||
		result.WorkloadDigest != benchmarkWorkloadDigest(result.PromptSuite) {
		return errors.New("benchmark: prompt suite or workload digest is unbound")
	}
	for _, prompt := range result.PromptSuite {
		if prompt == "" {
			return errors.New("benchmark: empty prompt in suite")
		}
	}
	if result.TokensPerSequence < minBenchmarkTokens || result.TokensPerSequence > maxBenchmarkTokens ||
		result.RequestedRuns < minBenchmarkRuns || result.RequestedRuns > maxBenchmarkRuns || len(result.Runs) != result.RequestedRuns ||
		result.RequestedWarmup < minBenchmarkWarmup || result.RequestedWarmup > maxBenchmarkWarmup || len(result.WarmupRuns) != result.RequestedWarmup ||
		result.BatchSequences < 0 || result.BatchSequences > maxBenchmarkSequences {
		return errors.New("benchmark: incomplete declared run or token denominator")
	}
	if result.DeviceTopK && (result.BatchSequences == 0 || result.Temperature == 0 || result.TopK == 0) ||
		result.BatchSequences > 0 && (result.CachePrompt || result.Speculative) {
		return errors.New("benchmark: unsupported execution configuration")
	}
	expected := result.TokensPerSequence
	if result.BatchSequences > 0 {
		expected *= result.BatchSequences
	}
	validateRun := func(run runMetrics, index int, prompt string) error {
		sequences := 1
		if result.BatchSequences > 0 {
			sequences = result.BatchSequences
		}
		if run.Run != index || run.PromptDigest != benchmarkPromptDigest(prompt) ||
			len(run.OutputDigests) != sequences {
			return fmt.Errorf("benchmark: run %d lacks exact prompt or sequence identity", index)
		}
		for _, digest := range run.OutputDigests {
			if len(digest) != 64 {
				return fmt.Errorf("benchmark: run %d has invalid token digest", index)
			}
			if _, err := hex.DecodeString(digest); err != nil {
				return fmt.Errorf("benchmark: run %d has invalid token digest: %w", index, err)
			}
		}
		if run.PromptTokens <= 0 || run.OutputTokens != expected || run.HostLogitTokens < 0 ||
			run.DeviceSelectedTokens < 0 || run.DeviceTopKTokens < 0 ||
			run.HostLogitTokens+run.DeviceSelectedTokens+run.DeviceTopKTokens != run.OutputTokens {
			return fmt.Errorf("benchmark: run %d has incomplete output or selection-path evidence", index)
		}
		if run.TotalMilliseconds <= 0 || math.IsNaN(run.TotalMilliseconds) || math.IsInf(run.TotalMilliseconds, 0) ||
			run.TTFTMilliseconds < 0 || run.TTFTMilliseconds > run.TotalMilliseconds {
			return fmt.Errorf("benchmark: run %d has invalid timing", index)
		}
		if result.Temperature > 0 && run.DeviceSelectedTokens != 0 ||
			!result.DeviceTopK && run.DeviceTopKTokens != 0 ||
			result.DeviceTopK && run.DeviceTopKTokens != expected {
			return fmt.Errorf("benchmark: run %d selection path differs from its sampling configuration", index)
		}
		return nil
	}
	for index, run := range result.WarmupRuns {
		if err := validateRun(run, -(index + 1), result.PromptSuite[0]); err != nil {
			return err
		}
	}
	for index, run := range result.Runs {
		if err := validateRun(run, index, result.PromptSuite[index%len(result.PromptSuite)]); err != nil {
			return err
		}
	}
	return nil
}
