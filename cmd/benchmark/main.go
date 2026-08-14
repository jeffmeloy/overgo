package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/cuda/driver"
	"overgo/internal/inference"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

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
	Device         int
	Tokens         int
	Runs           int
	Warmup         int
	CachePrompt    bool
	BatchSequences int
	ContextShift   bool
	Temperature    float64
	TopK           int
	DeviceTopK     bool
	LoRA           []string
}

type runMetrics struct {
	Run                      int     `json:"run"`
	PromptTokens             int     `json:"prompt_tokens"`
	CachedPromptTokens       int     `json:"cached_prompt_tokens"`
	OutputTokens             int     `json:"output_tokens"`
	PromptMilliseconds       float64 `json:"prompt_ms"`
	PromptTokensPerSecond    float64 `json:"prompt_tokens_per_second"`
	TTFTMilliseconds         float64 `json:"ttft_ms"`
	TotalMilliseconds        float64 `json:"total_ms"`
	DecodeMilliseconds       float64 `json:"decode_ms"`
	EndToEndTokensPerSecond  float64 `json:"end_to_end_tokens_per_second"`
	DecodeTokensPerSecond    float64 `json:"decode_tokens_per_second"`
	KernelLaunches           uint64  `json:"custom_kernel_launches"`
	StreamSynchronizations   uint64  `json:"stream_synchronizations"`
	HostToDeviceCopies       uint64  `json:"host_to_device_copies"`
	HostToDeviceBytes        uint64  `json:"host_to_device_bytes"`
	DeviceToHostCopies       uint64  `json:"device_to_host_copies"`
	DeviceToHostBytes        uint64  `json:"device_to_host_bytes"`
	DeviceToDeviceCopies     uint64  `json:"device_to_device_copies"`
	DeviceToDeviceBytes      uint64  `json:"device_to_device_bytes"`
	DeviceMemsets            uint64  `json:"device_memsets"`
	DeviceMemsetBytes        uint64  `json:"device_memset_bytes"`
	GraphInstantiations      uint64  `json:"graph_instantiations"`
	GraphUpdates             uint64  `json:"graph_updates"`
	GraphLaunches            uint64  `json:"graph_launches"`
	KernelLaunchesPerToken   float64 `json:"custom_kernel_launches_per_output_token"`
	SynchronizationsPerToken float64 `json:"stream_synchronizations_per_output_token"`
}

type summaryMetrics struct {
	TTFTMillisecondsP50        float64 `json:"ttft_ms_p50"`
	TTFTMillisecondsMinimum    float64 `json:"ttft_ms_min"`
	TotalMillisecondsP50       float64 `json:"total_ms_p50"`
	DecodeTokensPerSecondP50   float64 `json:"decode_tokens_per_second_p50"`
	EndToEndTokensPerSecondP50 float64 `json:"end_to_end_tokens_per_second_p50"`
}

type benchmarkResult struct {
	ModelPath              string                 `json:"model_path"`
	ModelName              string                 `json:"model_name"`
	Architecture           string                 `json:"architecture"`
	FileType               string                 `json:"file_type"`
	ParameterCount         uint64                 `json:"parameter_count"`
	ModelBytes             uint64                 `json:"model_bytes"`
	Residency              recipe.ResidencyPolicy `json:"residency"`
	CachePrompt            bool                   `json:"cache_prompt"`
	BatchSequences         int                    `json:"batch_sequences"`
	Temperature            float64                `json:"temperature"`
	TopK                   int                    `json:"top_k"`
	DeviceTopK             bool                   `json:"device_top_k"`
	Device                 driver.DeviceInfo      `json:"device"`
	LoadMilliseconds       float64                `json:"load_ms"`
	HostHeapBeforeBytes    uint64                 `json:"host_heap_before_bytes"`
	HostHeapAfterLoadBytes uint64                 `json:"host_heap_after_load_bytes"`
	HostHeapAfterRunsBytes uint64                 `json:"host_heap_after_runs_bytes"`
	DeviceAfterLoadBytes   uint64                 `json:"device_after_load_bytes"`
	DeviceAfterRunsBytes   uint64                 `json:"device_after_runs_bytes"`
	DevicePeakBytes        uint64                 `json:"device_peak_bytes"`
	Runs                   []runMetrics           `json:"runs"`
	Summary                summaryMetrics         `json:"summary"`
}

func main() {
	clioptions.Main(func() error { return run(os.Args[1:]) })
}

func parseOptions(args []string) (options, error) {
	flags := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	var result options
	modelFlags := clioptions.AddModelFlags(flags, "load GGUF LoRA adapter at scale 1; repeatable")
	flags.IntVar(&result.Tokens, "tokens", defaultBenchmarkTokens, "maximum generated tokens per run")
	flags.IntVar(&result.Runs, "runs", defaultBenchmarkRuns, "measured runs")
	flags.IntVar(&result.Warmup, "warmup", defaultBenchmarkWarmup, "unmeasured warmup runs")
	flags.BoolVar(&result.ContextShift, "context-shift", false, "enable rolling context shift")
	flags.Float64Var(&result.Temperature, "temperature", 0, "sampling temperature")
	flags.IntVar(&result.TopK, "top-k", 40, "sampling top-K limit")
	flags.BoolVar(&result.DeviceTopK, "device-top-k", false, "transfer bounded top-K candidates")
	flags.BoolVar(&result.CachePrompt, "cache-prompt", false, "reuse retained prompt state between runs")
	flags.IntVar(&result.BatchSequences, "batch-sequences", 0, "continuous-batch sequence count; zero uses Generate")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	result.Device = *modelFlags.DeviceOrdinal
	result.Repository = *modelFlags.Repository
	result.LoRA = modelFlags.LoRAPaths()
	if flags.NArg() != 2 {
		return options{}, errors.New("usage: benchmark [options] <model.gguf> <prompt>")
	}
	result.Model, result.Prompt = flags.Arg(0), flags.Arg(1)
	if result.Prompt == "" {
		return options{}, errors.New("benchmark: prompt must not be empty")
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
	if result.Temperature < 0 || result.TopK < 0 {
		return options{}, errors.New("benchmark: temperature and top-K must be non-negative")
	}
	if result.DeviceTopK && (result.BatchSequences == 0 || result.Temperature == 0 || result.TopK == 0) {
		return options{}, errors.New("benchmark: -device-top-k requires continuous sampling with positive temperature and top-K")
	}
	return result, nil
}

func run(args []string) error {
	options, err := parseOptions(args)
	if err != nil {
		return err
	}
	cuda, err := driver.Open()
	if err != nil {
		return err
	}
	defer cuda.Close()
	if err := cuda.Init(); err != nil {
		return err
	}
	device, err := cuda.DeviceInfo(options.Device)
	if err != nil {
		return err
	}
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	loadStarted := time.Now()
	openOptions := clioptions.BuildOpenOptions(options.Device, options.LoRA, 1)
	runner, err := clioptions.OpenRunner(
		context.Background(), options.Repository, options.Model, openOptions,
	)
	if err != nil {
		return err
	}
	defer runner.Close()
	loadDuration := time.Since(loadStarted)
	var afterLoad runtime.MemStats
	runtime.ReadMemStats(&afterLoad)
	deviceAfterLoad, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		return err
	}
	promptIDs, err := runner.TokenizeText(options.Prompt, true, false)
	if err != nil {
		return err
	}
	execute := func(index int) (runMetrics, error) {
		if options.BatchSequences > 0 {
			return executeContinuousBatch(context.Background(), runner, options, promptIDs, index)
		}
		sampler, samplerErr := sampling.New(sampling.Config{Temperature: float32(options.Temperature), TopK: options.TopK})
		if samplerErr != nil {
			return runMetrics{}, samplerErr
		}
		beforeExecution, statsErr := runner.DeviceExecutionStats(context.Background())
		if statsErr != nil {
			return runMetrics{}, statsErr
		}
		started := time.Now()
		firstToken := time.Time{}
		promptEvaluation := inference.PromptEvaluation{}
		outputTokens := 0
		_, _, generationErr := runner.Generate(context.Background(), options.Prompt, inference.GenerateOptions{
			MaxNewTokens: options.Tokens,
			Sampler:      sampler,
			// benchmark OnToken only counts; logits omission is acceptable
			DeviceGreedy: options.Temperature == 0,
			ContextShift: options.ContextShift,
			CachePrompt:  options.CachePrompt,
			OnPromptEvaluated: func(evaluation inference.PromptEvaluation) {
				promptEvaluation = evaluation
			},
			OnToken: func(inference.TokenEvent) error {
				outputTokens++
				if firstToken.IsZero() {
					firstToken = time.Now()
				}
				return nil
			},
		})
		finished := time.Now()
		if generationErr != nil {
			return runMetrics{}, generationErr
		}
		afterExecution, statsErr := runner.DeviceExecutionStats(context.Background())
		if statsErr != nil {
			return runMetrics{}, statsErr
		}
		execution := subtractExecutionStats(afterExecution, beforeExecution)
		total := finished.Sub(started)
		ttft := total
		if !firstToken.IsZero() {
			ttft = firstToken.Sub(started)
		}
		decode := total - ttft
		metrics := runMetrics{
			Run:                    index,
			PromptTokens:           len(promptIDs),
			CachedPromptTokens:     promptEvaluation.Cached,
			OutputTokens:           outputTokens,
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
	for index := 0; index < options.Warmup; index++ {
		if _, err := execute(-(index + 1)); err != nil {
			return fmt.Errorf("benchmark warmup %d: %w", index, err)
		}
	}
	runs := make([]runMetrics, options.Runs)
	for index := range runs {
		runs[index], err = execute(index)
		if err != nil {
			return fmt.Errorf("benchmark run %d: %w", index, err)
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
		ModelPath:              properties.Path,
		ModelName:              properties.Name,
		Architecture:           properties.Architecture,
		FileType:               properties.FileType,
		ParameterCount:         properties.ParameterCount,
		ModelBytes:             properties.ModelSize,
		Residency:              runner.Residency(),
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
	return clioptions.WritePrettyJSON(os.Stdout, result)
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
	started := time.Now()
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
	sampler, err := sampling.New(sampling.Config{Temperature: float32(options.Temperature), TopK: options.TopK})
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
	firstToken := time.Now()
	outputTokens := len(outputs)
	for generated := 1; generated < options.Tokens; generated++ {
		for sequence, output := range outputs {
			token, selectErr := selected(output)
			if selectErr != nil {
				return runMetrics{}, selectErr
			}
			inputs[sequence].Tokens = []tokenizer.TokenID{token}
		}
		outputs, err = step(ctx, inputs)
		if err != nil {
			return runMetrics{}, err
		}
		outputTokens += len(outputs)
	}
	finished := time.Now()
	afterExecution, err := runner.DeviceExecutionStats(ctx)
	if err != nil {
		return runMetrics{}, err
	}
	execution := subtractExecutionStats(afterExecution, beforeExecution)
	total := finished.Sub(started)
	ttft := firstToken.Sub(started)
	decode := total - ttft
	promptTokens := len(prompt) * options.BatchSequences
	metrics := runMetrics{
		Run: index, PromptTokens: promptTokens, OutputTokens: outputTokens,
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
	decode := make([]float64, len(runs))
	endToEnd := make([]float64, len(runs))
	for index, run := range runs {
		ttft[index] = run.TTFTMilliseconds
		total[index] = run.TotalMilliseconds
		decode[index] = run.DecodeTokensPerSecond
		endToEnd[index] = run.EndToEndTokensPerSecond
	}
	sort.Float64s(ttft)
	sort.Float64s(total)
	sort.Float64s(decode)
	sort.Float64s(endToEnd)
	return summaryMetrics{
		TTFTMillisecondsP50:        median(ttft),
		TTFTMillisecondsMinimum:    ttft[0],
		TotalMillisecondsP50:       median(total),
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
