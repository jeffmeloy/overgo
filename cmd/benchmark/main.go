//overgo:runtime-inputs caller

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/driver"
	"overgo/internal/inference"
	"overgo/internal/model"
	"overgo/internal/modelcli"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// Benchmark admission bounds are owned here so every CLI path rejects the
// same unsafe workload envelope before allocating runtime resources.
const (
	defaultBenchmarkTokens = 32
	minBenchmarkTokens     = 1
	maxBenchmarkTokens     = 1 << 20
	defaultBenchmarkRuns   = 5
	minBenchmarkRuns       = 1
	maxBenchmarkRuns       = 1000
	defaultBenchmarkWarmup = 1
	minBenchmarkWarmup     = 0
	maxBenchmarkWarmup     = 100
	maxBenchmarkSequences  = 1024
)

type options struct {
	Model          string
	Repository     string
	Prompt         string
	PromptSuite    string
	CompareFactor  string
	Device         int
	Tokens         int
	Runs           int
	Warmup         int
	CachePrompt    bool
	BatchSequences int
	ContextShift   bool
	Speculative    bool
	Temperature    float64
	TopK           int
	DeviceTopK     bool
	LoRA           []string
	Publish        bool
	CPUProfile     string
}

type benchmarkOptions = options

type runMetrics struct {
	Run                      int      `json:"run"`
	PromptDigest             string   `json:"prompt_digest"`
	OutputDigests            []string `json:"output_digests"`
	PromptTokens             int      `json:"prompt_tokens"`
	CachedPromptTokens       int      `json:"cached_prompt_tokens"`
	OutputTokens             int      `json:"output_tokens"`
	HostLogitTokens          int      `json:"host_logit_tokens"`
	DeviceSelectedTokens     int      `json:"device_selected_tokens"`
	DeviceTopKTokens         int      `json:"device_top_k_tokens"`
	PromptMilliseconds       float64  `json:"prompt_ms"`
	PromptTokensPerSecond    float64  `json:"prompt_tokens_per_second"`
	TTFTMilliseconds         float64  `json:"ttft_ms"`
	TotalMilliseconds        float64  `json:"total_ms"`
	DecodeMilliseconds       float64  `json:"decode_ms"`
	EndToEndTokensPerSecond  float64  `json:"end_to_end_tokens_per_second"`
	DecodeTokensPerSecond    float64  `json:"decode_tokens_per_second"`
	KernelLaunches           uint64   `json:"custom_kernel_launches"`
	StreamSynchronizations   uint64   `json:"stream_synchronizations"`
	HostToDeviceCopies       uint64   `json:"host_to_device_copies"`
	HostToDeviceBytes        uint64   `json:"host_to_device_bytes"`
	DeviceToHostCopies       uint64   `json:"device_to_host_copies"`
	DeviceToHostBytes        uint64   `json:"device_to_host_bytes"`
	DeviceToDeviceCopies     uint64   `json:"device_to_device_copies"`
	DeviceToDeviceBytes      uint64   `json:"device_to_device_bytes"`
	DeviceMemsets            uint64   `json:"device_memsets"`
	DeviceMemsetBytes        uint64   `json:"device_memset_bytes"`
	GraphInstantiations      uint64   `json:"graph_instantiations"`
	GraphUpdates             uint64   `json:"graph_updates"`
	GraphLaunches            uint64   `json:"graph_launches"`
	KernelLaunchesPerToken   float64  `json:"custom_kernel_launches_per_output_token"`
	SynchronizationsPerToken float64  `json:"stream_synchronizations_per_output_token"`
}

type summaryMetrics struct {
	TTFTMillisecondsP50        float64 `json:"ttft_ms_p50"`
	TTFTMillisecondsMinimum    float64 `json:"ttft_ms_min"`
	TotalMillisecondsP50       float64 `json:"total_ms_p50"`
	PromptTokensPerSecondP50   float64 `json:"prompt_tokens_per_second_p50"`
	DecodeTokensPerSecondP50   float64 `json:"decode_tokens_per_second_p50"`
	EndToEndTokensPerSecondP50 float64 `json:"end_to_end_tokens_per_second_p50"`
}

type benchmarkResult struct {
	ModelID            artifact.ID              `json:"model_id"`
	RecipeID           artifact.ID              `json:"recipe_id"`
	TokenizerContainer artifact.ID              `json:"tokenizer_container"`
	Environment        runrecord.Environment    `json:"environment"`
	EnvironmentID      artifact.ID              `json:"environment_id"`
	CodeCommit         string                   `json:"code_commit"`
	ModuleDigest       string                   `json:"module_digest"`
	OptionsDigest      string                   `json:"options_digest"`
	AdapterIDs         []artifact.ID            `json:"adapter_ids,omitempty"`
	ModelPath          string                   `json:"model_path"`
	ModelName          string                   `json:"model_name"`
	Architecture       string                   `json:"architecture"`
	FileType           string                   `json:"file_type"`
	ParameterCount     uint64                   `json:"parameter_count"`
	ModelBytes         uint64                   `json:"model_bytes"`
	Residency          recipe.ResidencyPolicy   `json:"residency"`
	RealizedResidency  recipe.RealizedResidency `json:"realized_residency"`
	// EndOfSequenceIgnored records continuation past EOG. SamplingProtocol
	// distinguishes this from historical results that banned EOG winners.
	EndOfSequenceIgnored   bool                    `json:"end_of_sequence_ignored"`
	SamplingProtocol       string                  `json:"sampling_protocol"`
	SamplerOrder           []sampling.SamplerStage `json:"sampler_order"`
	Prompt                 string                  `json:"prompt"`
	PromptSuite            []string                `json:"prompt_suite"`
	WorkloadDigest         string                  `json:"workload_digest"`
	RequestedWarmup        int                     `json:"requested_warmup"`
	WarmupRuns             []runMetrics            `json:"warmup_runs,omitempty"`
	TokensPerSequence      int                     `json:"tokens_per_sequence"`
	RequestedRuns          int                     `json:"requested_runs"`
	CachePrompt            bool                    `json:"cache_prompt"`
	BatchSequences         int                     `json:"batch_sequences"`
	Speculative            bool                    `json:"speculative"`
	ContextShift           bool                    `json:"context_shift"`
	Temperature            float64                 `json:"temperature"`
	TopK                   int                     `json:"top_k"`
	DeviceTopK             bool                    `json:"device_top_k"`
	Device                 driver.DeviceInfo       `json:"device"`
	LoadMilliseconds       float64                 `json:"load_ms"`
	HostHeapBeforeBytes    uint64                  `json:"host_heap_before_bytes"`
	HostHeapAfterLoadBytes uint64                  `json:"host_heap_after_load_bytes"`
	HostHeapAfterRunsBytes uint64                  `json:"host_heap_after_runs_bytes"`
	DeviceAfterLoadBytes   uint64                  `json:"device_after_load_bytes"`
	DeviceAfterRunsBytes   uint64                  `json:"device_after_runs_bytes"`
	DevicePeakBytes        uint64                  `json:"device_peak_bytes"`
	Runs                   []runMetrics            `json:"runs"`
	Summary                summaryMetrics          `json:"summary"`
}

func main() {
	clioptions.Main(func() error { return run(os.Args[1:]) })
}

func parseOptions(args []string) (options, error) {
	flags := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	var result options
	modelFlags := modelcli.AddModelFlags(flags, "load GGUF LoRA adapter at scale 1; repeatable")
	flags.StringVar(&result.PromptSuite, "prompt-suite", "", "JSON array of prompts, rotated in order across measured runs")
	flags.StringVar(&result.CompareFactor, "compare", "", "interleaved lossless comparison factor: cache_prompt or speculative")
	flags.IntVar(&result.Tokens, "tokens", defaultBenchmarkTokens, "maximum generated tokens per run")
	flags.IntVar(&result.Runs, "runs", defaultBenchmarkRuns, "measured runs")
	flags.IntVar(&result.Warmup, "warmup", defaultBenchmarkWarmup, "unmeasured warmup runs")
	flags.BoolVar(&result.ContextShift, "context-shift", false, "enable rolling context shift")
	flags.BoolVar(&result.Speculative, "speculative", false, "enable NextN MTP speculative decode")
	flags.Float64Var(&result.Temperature, "temperature", 0, "sampling temperature")
	flags.IntVar(&result.TopK, "top-k", 40, "sampling top-K limit")
	flags.BoolVar(&result.DeviceTopK, "device-top-k", false, "transfer bounded top-K candidates")
	flags.BoolVar(&result.CachePrompt, "cache-prompt", false, "reuse retained prompt state between runs")
	flags.IntVar(&result.BatchSequences, "batch-sequences", 0, "continuous-batch sequence count; zero uses Generate")
	flags.BoolVar(&result.Publish, "publish", false, "commit the result as benchmark evidence with a verification claim (requires -repo and a clean worktree)")
	flags.StringVar(&result.CPUProfile, "cpuprofile", "", "write a Go CPU profile of the whole run to this file (host-side time between kernels)")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	result.Device = *modelFlags.DeviceOrdinal
	result.Repository = *modelFlags.Repository
	result.LoRA = modelFlags.LoRAPaths()
	if result.PromptSuite == "" {
		if flags.NArg() != 2 || flags.Arg(1) == "" {
			return options{}, errors.New("usage: benchmark [options] <model.gguf> <prompt>")
		}
		result.Model, result.Prompt = flags.Arg(0), flags.Arg(1)
	} else {
		if flags.NArg() != 1 {
			return options{}, errors.New("usage: benchmark -prompt-suite prompts.json [options] <model.gguf>")
		}
		result.Model = flags.Arg(0)
	}
	if result.Tokens < minBenchmarkTokens || result.Tokens > maxBenchmarkTokens {
		return options{}, fmt.Errorf("benchmark: -tokens must be in [%d,%d]", minBenchmarkTokens, maxBenchmarkTokens)
	}
	if result.Runs < minBenchmarkRuns || result.Runs > maxBenchmarkRuns {
		return options{}, fmt.Errorf("benchmark: -runs must be in [%d,%d]", minBenchmarkRuns, maxBenchmarkRuns)
	}
	if result.Warmup < minBenchmarkWarmup || result.Warmup > maxBenchmarkWarmup {
		return options{}, fmt.Errorf("benchmark: -warmup must be in [%d,%d]", minBenchmarkWarmup, maxBenchmarkWarmup)
	}
	if result.BatchSequences < 0 || result.BatchSequences > maxBenchmarkSequences {
		return options{}, fmt.Errorf("benchmark: -batch-sequences must be in [0,%d]", maxBenchmarkSequences)
	}
	if result.BatchSequences > 0 && result.CachePrompt {
		return options{}, errors.New("benchmark: -cache-prompt is unavailable in continuous-batch mode")
	}
	if result.BatchSequences > 0 && result.Speculative {
		return options{}, errors.New("benchmark: speculative decoding is unavailable in continuous-batch mode")
	}
	if result.Temperature < 0 || result.TopK < 0 {
		return options{}, errors.New("benchmark: temperature and top-K must be non-negative")
	}
	if result.DeviceTopK && (result.BatchSequences == 0 || result.Temperature == 0 || result.TopK == 0) {
		return options{}, errors.New("benchmark: -device-top-k requires continuous sampling with positive temperature and top-K")
	}
	if result.CompareFactor != "" {
		if result.Publish || result.BatchSequences != 0 || result.Temperature != 0 ||
			result.CompareFactor != "cache_prompt" && result.CompareFactor != "speculative" ||
			result.CompareFactor == "cache_prompt" && result.CachePrompt ||
			result.CompareFactor == "speculative" && result.Speculative {
			return options{}, errors.New("benchmark: paired comparison requires one disabled lossless factor, single-sequence greedy mode, and no direct publication")
		}
	}
	if result.Publish && result.Repository == "" {
		return options{}, errors.New("benchmark: -publish requires -repo so the evidence has a store to land in")
	}
	if _, err := benchmarkSampler(result); err != nil {
		return options{}, fmt.Errorf("benchmark sampling: %w", err)
	}
	return result, nil
}

func run(args []string) error {
	options, err := parseOptions(args)
	if err != nil {
		return err
	}
	prompts, err := benchmarkPrompts(options)
	if err != nil {
		return err
	}
	if remoteprovider.IsRemoteLocation(options.Model) {
		return errors.New("benchmark: a hosted model has no local decode to measure; evaluate scores it through the relay")
	}
	if options.CPUProfile != "" {
		profile, profileErr := os.Create(options.CPUProfile)
		if profileErr != nil {
			return profileErr
		}
		defer profile.Close()
		if profileErr := pprof.StartCPUProfile(profile); profileErr != nil {
			return profileErr
		}
		defer pprof.StopCPUProfile()
	}
	cuda, err := driver.Open()
	if err != nil {
		return err
	}
	defer cuda.Close()
	if err := cuda.Init(); err != nil {
		return err
	}
	device, err := cuda.ReserveDevice(options.Device)
	if err != nil {
		return err
	}
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	loadStarted := processmeasure.NewStopwatch()
	openOptions := modelcli.BuildOpenOptions(options.Device, options.LoRA, 1)
	runner, err := modelcli.OpenRunner(
		context.Background(), options.Repository, options.Model, openOptions,
	)
	if err != nil {
		return err
	}
	defer runner.Close()
	codeCommit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return fmt.Errorf("benchmark source identity: %w", err)
	}
	moduleDigest, err := benchmarkModuleDigest()
	if err != nil {
		return err
	}
	adapters, err := benchmarkAdapterIDs(options.LoRA)
	if err != nil {
		return fmt.Errorf("benchmark adapter identity: %w", err)
	}
	optionsDigest, err := benchmarkOptionsDigest(options, adapters)
	if err != nil {
		return err
	}
	description, err := runner.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil {
		return err
	}
	environment, err := runrecord.CurrentEnvironment(device.UUID, "cuda")
	if err != nil {
		return err
	}
	driverVersion, err := cuda.DriverVersion()
	if err != nil {
		return err
	}
	environment.Driver = driverVersion.String()
	environment, err = runrecord.NewEnvironment(environment)
	if err != nil {
		return err
	}
	loadDuration, err := loadStarted.Elapsed()
	if err != nil {
		return fmt.Errorf("benchmark load measurement: %w", err)
	}
	var afterLoad runtime.MemStats
	runtime.ReadMemStats(&afterLoad)
	deviceAfterLoad, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		return err
	}
	promptIDs := make([][]tokenizer.TokenID, len(prompts))
	for index, prompt := range prompts {
		promptIDs[index], err = runner.TokenizeText(prompt, true, false)
		if err != nil {
			return fmt.Errorf("benchmark prompt %d: %w", index, err)
		}
	}
	execute := func(index int, arm benchmarkOptions) (runMetrics, error) {
		promptIndex := benchmarkPromptIndex(index, len(prompts))
		prompt := prompts[promptIndex]
		ids := promptIDs[promptIndex]
		if arm.BatchSequences > 0 {
			return executeContinuousBatch(context.Background(), runner, arm, ids, prompt, index)
		}
		generation, samplerErr := benchmarkGenerationOptions(arm)
		if samplerErr != nil {
			return runMetrics{}, samplerErr
		}
		beforeExecution, statsErr := runner.DeviceExecutionStats(context.Background())
		if statsErr != nil {
			return runMetrics{}, statsErr
		}
		timing := startGenerationTiming()
		promptEvaluation := inference.PromptEvaluation{}
		outputTokens := 0
		outputDigest := newTokenDigest()
		hostLogitTokens, deviceSelectedTokens := 0, 0
		generation.OnPromptEvaluated = func(evaluation inference.PromptEvaluation) { promptEvaluation = evaluation }
		generation.OnToken = func(event inference.TokenEvent) error {
			outputTokens++
			outputDigest.add(event.ID)
			if len(event.Logits) == 0 {
				deviceSelectedTokens++
			} else {
				hostLogitTokens++
			}
			return timing.token()
		}
		_, _, generationErr := runner.Generate(context.Background(), prompt, generation)
		total, ttft, decode, timingErr := timing.finish()
		if generationErr != nil {
			return runMetrics{}, generationErr
		}
		if timingErr != nil {
			return runMetrics{}, timingErr
		}
		afterExecution, statsErr := runner.DeviceExecutionStats(context.Background())
		if statsErr != nil {
			return runMetrics{}, statsErr
		}
		execution := subtractExecutionStats(afterExecution, beforeExecution)
		metrics := runMetrics{
			Run:                    index,
			PromptDigest:           benchmarkPromptDigest(prompt),
			OutputDigests:          []string{outputDigest.sum()},
			PromptTokens:           len(ids),
			CachedPromptTokens:     promptEvaluation.Cached,
			OutputTokens:           outputTokens,
			HostLogitTokens:        hostLogitTokens,
			DeviceSelectedTokens:   deviceSelectedTokens,
			PromptMilliseconds:     float64(promptEvaluation.Duration) / float64(time.Millisecond),
			TTFTMilliseconds:       float64(ttft) / float64(time.Millisecond),
			TotalMilliseconds:      float64(total) / float64(time.Millisecond),
			DecodeMilliseconds:     float64(decode) / float64(time.Millisecond),
			KernelLaunches:         execution.KernelLaunches,
			StreamSynchronizations: execution.StreamSynchronizations,
			HostToDeviceCopies:     execution.HostToDeviceCopies,
			HostToDeviceBytes:      execution.HostToDeviceBytes,
			DeviceToHostCopies:     execution.DeviceToHostCopies,
			DeviceToHostBytes:      execution.DeviceToHostBytes,
			DeviceToDeviceCopies:   execution.DeviceToDeviceCopies,
			DeviceToDeviceBytes:    execution.DeviceToDeviceBytes,
			DeviceMemsets:          execution.DeviceMemsets,
			DeviceMemsetBytes:      execution.DeviceMemsetBytes,
			GraphInstantiations:    execution.GraphInstantiations,
			GraphUpdates:           execution.GraphUpdates,
			GraphLaunches:          execution.GraphLaunches,
		}
		uncachedPromptTokens := promptEvaluation.Tokens - promptEvaluation.Cached
		if uncachedPromptTokens > 0 && promptEvaluation.Duration > 0 {
			metrics.PromptTokensPerSecond = float64(uncachedPromptTokens) / promptEvaluation.Duration.Seconds()
		}
		if outputTokens > 0 {
			metrics.KernelLaunchesPerToken = float64(execution.KernelLaunches) / float64(outputTokens)
			metrics.SynchronizationsPerToken = float64(execution.StreamSynchronizations) / float64(outputTokens)
		}
		if total > 0 {
			metrics.EndToEndTokensPerSecond = float64(outputTokens) / total.Seconds()
		}
		if outputTokens > 1 && decode > 0 {
			metrics.DecodeTokensPerSecond = float64(outputTokens-1) / decode.Seconds()
		}
		return metrics, nil
	}
	var candidate benchmarkOptions
	var candidateWarmups, candidateRuns []runMetrics
	var pairFirst []string
	if options.CompareFactor != "" {
		candidate = options
		switch options.CompareFactor {
		case "cache_prompt":
			candidate.CachePrompt = true
		case "speculative":
			candidate.Speculative = true
		}
		candidateWarmups = make([]runMetrics, options.Warmup)
		candidateRuns = make([]runMetrics, options.Runs)
		pairFirst = make([]string, options.Runs)
	}
	warmups := make([]runMetrics, options.Warmup)
	for index := range warmups {
		warmups[index], err = execute(-(index + 1), options)
		if err != nil {
			return fmt.Errorf("benchmark baseline warmup %d: %w", index, err)
		}
		if candidateWarmups != nil {
			candidateWarmups[index], err = execute(-(index + 1), candidate)
			if err != nil {
				return fmt.Errorf("benchmark candidate warmup %d: %w", index, err)
			}
		}
	}
	runs := make([]runMetrics, options.Runs)
	for index := range runs {
		if candidateRuns != nil && index%2 == 1 {
			pairFirst[index] = "candidate"
			candidateRuns[index], err = execute(index, candidate)
			if err != nil {
				return fmt.Errorf("benchmark candidate run %d: %w", index, err)
			}
		}
		runs[index], err = execute(index, options)
		if err != nil {
			return fmt.Errorf("benchmark baseline run %d: %w", index, err)
		}
		if candidateRuns != nil && index%2 == 0 {
			pairFirst[index] = "baseline"
			candidateRuns[index], err = execute(index, candidate)
			if err != nil {
				return fmt.Errorf("benchmark candidate run %d: %w", index, err)
			}
		}
	}
	var afterRuns runtime.MemStats
	runtime.ReadMemStats(&afterRuns)
	deviceAfterRuns, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		return err
	}
	properties := runner.ModelProperties()
	result := benchmarkResult{
		ModelID:                runner.ModelID(),
		RecipeID:               description.Identity.Recipe,
		TokenizerContainer:     runner.ModelID(),
		Environment:            environment,
		EnvironmentID:          environment.ID,
		CodeCommit:             codeCommit,
		ModuleDigest:           moduleDigest,
		OptionsDigest:          optionsDigest,
		AdapterIDs:             adapters,
		ModelPath:              properties.Path,
		ModelName:              properties.Name,
		Architecture:           properties.Architecture,
		FileType:               properties.FileType,
		ParameterCount:         properties.ParameterCount,
		ModelBytes:             properties.ModelSize,
		Residency:              runner.Residency(),
		RealizedResidency:      runner.RealizedResidency(),
		EndOfSequenceIgnored:   true,
		SamplingProtocol:       benchmarkProtocol(options.Temperature),
		SamplerOrder:           benchmarkSamplingConfig(options).Samplers,
		Prompt:                 prompts[0],
		PromptSuite:            prompts,
		WorkloadDigest:         benchmarkWorkloadDigest(prompts),
		RequestedWarmup:        options.Warmup,
		WarmupRuns:             warmups,
		TokensPerSequence:      options.Tokens,
		RequestedRuns:          options.Runs,
		Speculative:            options.Speculative,
		ContextShift:           options.ContextShift,
		CachePrompt:            options.CachePrompt,
		BatchSequences:         options.BatchSequences,
		Temperature:            options.Temperature,
		TopK:                   options.TopK,
		DeviceTopK:             options.DeviceTopK,
		Device:                 device,
		LoadMilliseconds:       float64(loadDuration) / float64(time.Millisecond),
		HostHeapBeforeBytes:    before.HeapAlloc,
		HostHeapAfterLoadBytes: afterLoad.HeapAlloc,
		HostHeapAfterRunsBytes: afterRuns.HeapAlloc,
		DeviceAfterLoadBytes:   deviceAfterLoad.CurrentBytes,
		DeviceAfterRunsBytes:   deviceAfterRuns.CurrentBytes,
		DevicePeakBytes:        deviceAfterRuns.PeakBytes,
		Runs:                   runs,
		Summary:                summarizeRuns(runs),
	}
	if err := validateBenchmarkResult(result); err != nil {
		return err
	}
	if candidateRuns != nil {
		candidateResult := result
		candidateResult.Runs = candidateRuns
		candidateResult.WarmupRuns = candidateWarmups
		candidateResult.Summary = summarizeRuns(candidateRuns)
		candidateResult.CachePrompt = candidate.CachePrompt
		candidateResult.Speculative = candidate.Speculative
		candidateResult.OptionsDigest, err = benchmarkOptionsDigest(candidate, adapters)
		if err != nil {
			return err
		}
		comparison, compareErr := compareBenchmarkResults(result, candidateResult, options.CompareFactor, pairFirst)
		if compareErr != nil {
			return compareErr
		}
		return clioptions.WritePrettyJSON(os.Stdout, pairedBenchmarkResult{
			Baseline: result, Candidate: candidateResult, Comparison: comparison, ComparisonID: comparison.ID,
		})
	}
	if err := clioptions.WritePrettyJSON(os.Stdout, result); err != nil {
		return err
	}
	if !options.Publish {
		return nil
	}
	return publishBenchmarkEvidence(context.Background(), options, result)
}

func subtractExecutionStats(after, before driver.ExecutionStats) driver.ExecutionStats {
	return driver.ExecutionStats{
		KernelLaunches:          after.KernelLaunches - before.KernelLaunches,
		StreamSynchronizations:  after.StreamSynchronizations - before.StreamSynchronizations,
		ContextSynchronizations: after.ContextSynchronizations - before.ContextSynchronizations,
		HostToDeviceCopies:      after.HostToDeviceCopies - before.HostToDeviceCopies,
		HostToDeviceBytes:       after.HostToDeviceBytes - before.HostToDeviceBytes,
		DeviceToHostCopies:      after.DeviceToHostCopies - before.DeviceToHostCopies,
		DeviceToHostBytes:       after.DeviceToHostBytes - before.DeviceToHostBytes,
		DeviceToDeviceCopies:    after.DeviceToDeviceCopies - before.DeviceToDeviceCopies,
		DeviceToDeviceBytes:     after.DeviceToDeviceBytes - before.DeviceToDeviceBytes,
		DeviceMemsets:           after.DeviceMemsets - before.DeviceMemsets,
		DeviceMemsetBytes:       after.DeviceMemsetBytes - before.DeviceMemsetBytes,
		GraphInstantiations:     after.GraphInstantiations - before.GraphInstantiations,
		GraphUpdates:            after.GraphUpdates - before.GraphUpdates,
		GraphLaunches:           after.GraphLaunches - before.GraphLaunches,
	}
}

func executeContinuousBatch(
	ctx context.Context,
	runner *inference.Runner,
	options options,
	prompt []tokenizer.TokenID,
	promptText string,
	index int,
) (runMetrics, error) {
	device := runner.DeviceResident()
	batch, err := runner.NewContinuousBatch(inference.ContinuousBatchOptions{
		MaxSequences: options.BatchSequences,
		Device:       device,
		ContextShift: options.ContextShift,
	})
	if err != nil {
		return runMetrics{}, err
	}
	defer batch.Close(context.Background())
	inputs := make([]inference.SequenceBatchInput, options.BatchSequences)
	for sequence := range inputs {
		inputs[sequence] = inference.SequenceBatchInput{ID: inference.SequenceID(sequence + 1), Tokens: prompt}
	}
	beforeExecution, err := runner.DeviceExecutionStats(ctx)
	if err != nil {
		return runMetrics{}, err
	}
	timing := startGenerationTiming()
	deviceGreedy := device && options.Temperature == 0 && runner.Spec().Profile().Attention == model.AttentionGatedDelta
	step := batch.Step
	if deviceGreedy {
		step = batch.StepGreedy
	} else if options.DeviceTopK {
		step = func(ctx context.Context, inputs []inference.SequenceBatchInput) ([]inference.SequenceBatchOutput, error) {
			return batch.StepTopK(ctx, inputs, uint32(options.TopK))
		}
	}
	outputs, err := step(ctx, inputs)
	if err != nil {
		return runMetrics{}, err
	}
	sampler, err := benchmarkSampler(options)
	if err != nil {
		return runMetrics{}, err
	}
	selected := func(output inference.SequenceBatchOutput) (tokenizer.TokenID, error) {
		if deviceGreedy {
			return output.Token, nil
		}
		if options.DeviceTopK {
			ids := make([]int, len(output.Candidates))
			logits := make([]float32, len(output.Candidates))
			for index, candidate := range output.Candidates {
				ids[index], logits[index] = int(candidate.ID), candidate.Logit
			}
			id, sampleErr := sampler.SampleTopK(ids, logits, runner.SamplingVocabularySize())
			return tokenizer.TokenID(id), sampleErr
		}
		id, sampleErr := sampler.Sample(output.Logits)
		return tokenizer.TokenID(id), sampleErr
	}
	outputTokens := 0
	outputDigests := make([]*tokenDigest, options.BatchSequences)
	for index := range outputDigests {
		outputDigests[index] = newTokenDigest()
	}
	for generated := range options.Tokens {
		if len(outputs) != len(inputs) {
			return runMetrics{}, errors.New("benchmark: incomplete batch output")
		}
		for sequence, output := range outputs {
			token, selectErr := selected(output)
			if selectErr != nil {
				return runMetrics{}, selectErr
			}
			inputs[sequence].Tokens = []tokenizer.TokenID{token}
			outputDigests[sequence].add(token)
			outputTokens++
		}
		if err := timing.token(); err != nil {
			return runMetrics{}, err
		}
		if generated+1 == options.Tokens {
			break
		}
		outputs, err = step(ctx, inputs)
		if err != nil {
			return runMetrics{}, err
		}
	}
	total, ttft, decode, err := timing.finish()
	if err != nil {
		return runMetrics{}, err
	}
	afterExecution, err := runner.DeviceExecutionStats(ctx)
	if err != nil {
		return runMetrics{}, err
	}
	execution := subtractExecutionStats(afterExecution, beforeExecution)
	promptTokens := len(prompt) * options.BatchSequences
	digests := make([]string, len(outputDigests))
	for index, digest := range outputDigests {
		digests[index] = digest.sum()
	}
	metrics := runMetrics{
		Run: index, PromptDigest: benchmarkPromptDigest(promptText), OutputDigests: digests,
		PromptTokens: promptTokens, OutputTokens: outputTokens,
		PromptMilliseconds: float64(ttft) / float64(time.Millisecond),
		TTFTMilliseconds:   float64(ttft) / float64(time.Millisecond),
		TotalMilliseconds:  float64(total) / float64(time.Millisecond),
		DecodeMilliseconds: float64(decode) / float64(time.Millisecond),
		KernelLaunches:     execution.KernelLaunches, StreamSynchronizations: execution.StreamSynchronizations,
		HostToDeviceCopies: execution.HostToDeviceCopies, HostToDeviceBytes: execution.HostToDeviceBytes,
		DeviceToHostCopies: execution.DeviceToHostCopies, DeviceToHostBytes: execution.DeviceToHostBytes,
		DeviceToDeviceCopies: execution.DeviceToDeviceCopies, DeviceToDeviceBytes: execution.DeviceToDeviceBytes,
		DeviceMemsets: execution.DeviceMemsets, DeviceMemsetBytes: execution.DeviceMemsetBytes,
		GraphInstantiations: execution.GraphInstantiations,
		GraphUpdates:        execution.GraphUpdates, GraphLaunches: execution.GraphLaunches,
	}
	switch {
	case deviceGreedy:
		metrics.DeviceSelectedTokens = outputTokens
	case options.DeviceTopK:
		metrics.DeviceTopKTokens = outputTokens
	default:
		metrics.HostLogitTokens = outputTokens
	}
	if ttft > 0 {
		metrics.PromptTokensPerSecond = float64(promptTokens) / ttft.Seconds()
	}
	if outputTokens > 0 {
		metrics.KernelLaunchesPerToken = float64(execution.KernelLaunches) / float64(outputTokens)
		metrics.SynchronizationsPerToken = float64(execution.StreamSynchronizations) / float64(outputTokens)
		metrics.EndToEndTokensPerSecond = float64(outputTokens) / total.Seconds()
	}
	decodeTokens := outputTokens - options.BatchSequences
	if decodeTokens > 0 && decode > 0 {
		metrics.DecodeTokensPerSecond = float64(decodeTokens) / decode.Seconds()
	}
	return metrics, nil
}

func summarizeRuns(runs []runMetrics) summaryMetrics {
	ttft := make([]float64, len(runs))
	total := make([]float64, len(runs))
	prompt := make([]float64, len(runs))
	decode := make([]float64, len(runs))
	endToEnd := make([]float64, len(runs))
	for index, run := range runs {
		ttft[index] = run.TTFTMilliseconds
		total[index] = run.TotalMilliseconds
		prompt[index] = run.PromptTokensPerSecond
		decode[index] = run.DecodeTokensPerSecond
		endToEnd[index] = run.EndToEndTokensPerSecond
	}
	slices.Sort(ttft)
	slices.Sort(total)
	slices.Sort(prompt)
	slices.Sort(decode)
	slices.Sort(endToEnd)
	return summaryMetrics{
		TTFTMillisecondsP50:        median(ttft),
		TTFTMillisecondsMinimum:    ttft[0],
		TotalMillisecondsP50:       median(total),
		PromptTokensPerSecondP50:   median(prompt),
		DecodeTokensPerSecondP50:   median(decode),
		EndToEndTokensPerSecondP50: median(endToEnd),
	}
}

func median(sorted []float64) float64 {
	middle := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[middle]
	}
	return (sorted[middle-1] + sorted[middle]) / 2
}
