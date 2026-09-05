package main

import (
	"errors"
	"fmt"
	"math"
	"slices"

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
	opts := options{Temperature: result.Temperature, TopK: result.TopK}
	if _, err := benchmarkSampler(opts); err != nil {
		return err
	}
	if !result.EndOfSequenceIgnored || result.SamplingProtocol != benchmarkProtocol(result.Temperature) ||
		!slices.Equal(result.SamplerOrder, benchmarkSamplingConfig(opts).Samplers) {
		return errors.New("benchmark: sampling protocol is absent or differs from the measured configuration")
	}
	if result.TokensPerSequence < minBenchmarkTokens || result.TokensPerSequence > maxBenchmarkTokens ||
		result.RequestedRuns < minBenchmarkRuns || result.RequestedRuns > maxBenchmarkRuns || len(result.Runs) != result.RequestedRuns ||
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
	for index, run := range result.Runs {
		if run.Run != index || run.PromptTokens <= 0 || run.OutputTokens != expected || run.HostLogitTokens < 0 ||
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
	}
	return nil
}
