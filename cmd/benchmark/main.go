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

	"llamacpp2go/internal/clioptions"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/sampling"
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
)

type options struct {
	Model        string
	Prompt       string
	Device       int
	Tokens       int
	Runs         int
	Warmup       int
	Preload      bool
	NativeQuant  bool
	ContextShift bool
	LoRA         []string
}

type runMetrics struct {
	Run                      int     `json:"run"`
	PromptTokens             int     `json:"prompt_tokens"`
	OutputTokens             int     `json:"output_tokens"`
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
	ModelPath              string            `json:"model_path"`
	ModelName              string            `json:"model_name"`
	Architecture           string            `json:"architecture"`
	FileType               string            `json:"file_type"`
	ParameterCount         uint64            `json:"parameter_count"`
	ModelBytes             uint64            `json:"model_bytes"`
	Device                 driver.DeviceInfo `json:"device"`
	LoadMilliseconds       float64           `json:"load_ms"`
	HostHeapBeforeBytes    uint64            `json:"host_heap_before_bytes"`
	HostHeapAfterLoadBytes uint64            `json:"host_heap_after_load_bytes"`
	HostHeapAfterRunsBytes uint64            `json:"host_heap_after_runs_bytes"`
	DeviceAfterLoadBytes   uint64            `json:"device_after_load_bytes"`
	DeviceAfterRunsBytes   uint64            `json:"device_after_runs_bytes"`
	DevicePeakBytes        uint64            `json:"device_peak_bytes"`
	Runs                   []runMetrics      `json:"runs"`
	Summary                summaryMetrics    `json:"summary"`
}

func main() {
	clioptions.Main(func() error { return run(os.Args[1:]) })
}

func parseOptions(args []string) (options, error) {
	flags := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	var result options
	modelFlags := clioptions.AddModelFlagsWithConfig(flags, "load GGUF LoRA adapter at scale 1; repeatable", clioptions.ModelFlagConfig{
		PreloadName: "preload", NativeQuantName: "native-quant",
	})
	flags.IntVar(&result.Tokens, "tokens", defaultBenchmarkTokens, "maximum generated tokens per run")
	flags.IntVar(&result.Runs, "runs", defaultBenchmarkRuns, "measured runs")
	flags.IntVar(&result.Warmup, "warmup", defaultBenchmarkWarmup, "unmeasured warmup runs")
	flags.BoolVar(&result.ContextShift, "context-shift", false, "enable rolling context shift")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	result.Device = *modelFlags.DeviceOrdinal
	result.Preload = modelFlags.Preload != nil && *modelFlags.Preload
	result.NativeQuant = modelFlags.NativeQuant != nil && *modelFlags.NativeQuant
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
	if result.Preload && result.NativeQuant {
		return options{}, errors.New("benchmark: -preload and -native-quant are mutually exclusive")
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
	runner, err := inference.OpenWithOptions(options.Model, clioptions.BuildOpenOptions(
		options.Device, options.Preload, options.NativeQuant, options.LoRA, 1,
	))
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
		sampler, samplerErr := sampling.New(sampling.Config{Temperature: 0})
		if samplerErr != nil {
			return runMetrics{}, samplerErr
		}
		beforeExecution, statsErr := runner.DeviceExecutionStats(context.Background())
		if statsErr != nil {
			return runMetrics{}, statsErr
		}
		started := time.Now()
		firstToken := time.Time{}
		outputTokens := 0
		_, _, generationErr := runner.Generate(context.Background(), options.Prompt, inference.GenerateOptions{
			MaxNewTokens: options.Tokens,
			Sampler:      sampler,
			ContextShift: options.ContextShift,
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
			OutputTokens:           outputTokens,
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
	}
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
