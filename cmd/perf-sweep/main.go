// Command perf-sweep measures one servable model's inference against the
// fixed SFT prompt fixture: prompt context, generation wall, and peak device
// memory. One model per process so the device high-water belongs to exactly
// one artifact. The report is a single JSON line; recording it as a
// verification claim stays with the compatibility record producer.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"overgo/internal/checked"
	"overgo/internal/inference"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
)

type promptFixture struct {
	Schema    string   `json:"schema"`
	Source    string   `json:"source"`
	Prompt    string   `json:"prompt"`
	Expected  []string `json:"expected_gpt"`
	MaxTokens int      `json:"max_tokens"`
}

type report struct {
	ModelPath       string `json:"model_path"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
	WallNS          uint64 `json:"wall_ns"`
	PeakDeviceBytes uint64 `json:"peak_device_bytes"`
	TextHead        string `json:"text_head"`
	FixtureSource   string `json:"fixture_source"`
	GoVersion       string `json:"go_version"`
	OSArch          string `json:"os_arch"`
}

type sweepBaseline struct {
	Schema                    string        `json:"schema"`
	SourceCommit              string        `json:"source_commit"`
	Fixture                   string        `json:"fixture"`
	RegressionScale           uint64        `json:"regression_scale"`
	MaximumWallIncrease       uint64        `json:"maximum_wall_increase"`
	MaximumPeakDeviceIncrease uint64        `json:"maximum_peak_device_increase"`
	Models                    []sweepTarget `json:"models"`
}

type sweepTarget struct {
	Name            string `json:"name"`
	ModelPath       string `json:"model_path"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
	WallNS          uint64 `json:"wall_ns"`
	PeakDeviceBytes uint64 `json:"peak_device_bytes"`
}

const (
	sweepBaselineSchema      = "overgo/performance-sweep-baseline/v1"
	performanceTextHeadBytes = 160
)

func main() {
	model := flag.String("model", "", "GGUF model path")
	prompts := flag.String("prompts", "fixtures/sweep_sft_prompts.json", "fixed SFT prompt fixture")
	check := flag.Bool("check", false, "run every installed baseline model and reject material wall or device-memory regressions")
	baseline := flag.String("baseline", "docs/performance_sweep_baseline.json", "performance sweep baseline used by -check")
	flag.Parse()
	usageMessage := ""
	var err error
	switch {
	case *check && (*model != "" || flag.NArg() != 0):
		usageMessage = "usage: perf-sweep -check [-baseline baseline.json] [-prompts fixture.json]"
	case *check:
		err = runCheck(*baseline, *prompts)
	case *model == "" || flag.NArg() != 0:
		usageMessage = "usage: perf-sweep -model <model.gguf> [-prompts fixture.json] | perf-sweep -check"
	default:
		err = run(*model, *prompts)
	}
	if usageMessage != "" {
		fmt.Fprintln(os.Stderr, usageMessage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "perf-sweep:", err)
		os.Exit(1)
	}
}

func run(modelPath, promptsPath string) error {
	result, err := measure(modelPath, promptsPath)
	if err != nil {
		return err
	}
	return writeReport(result)
}

func measure(modelPath, promptsPath string) (report, error) {
	var fixture promptFixture
	if err := jsonfile.Decode(promptsPath, &fixture); err != nil {
		return report{}, err
	}
	if fixture.Prompt == "" || fixture.MaxTokens <= 0 {
		return report{}, fmt.Errorf("fixture %s carries no prompt or budget", promptsPath)
	}
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative,
	)
	if err != nil {
		return report{}, err
	}
	runner, err := inference.OpenWithProgram(context.Background(), &loaded, inference.OpenOptions{})
	if err != nil {
		return report{}, err
	}
	defer runner.Close()
	var greedyConfig sampling.Config
	greedy, err := sampling.New(greedyConfig)
	if err != nil {
		return report{}, err
	}
	var text strings.Builder
	promptTokens := 0
	started := time.Now()
	ids, _, err := runner.Generate(context.Background(), fixture.Prompt, inference.GenerateOptions{
		MaxNewTokens: fixture.MaxTokens, Sampler: greedy, DeviceGreedy: true,
		OnToken: func(event inference.TokenEvent) error {
			text.WriteString(event.Piece)
			return nil
		},
		OnPromptEvaluated: func(evaluation inference.PromptEvaluation) {
			promptTokens = evaluation.Tokens
		},
	})
	if err != nil {
		return report{}, err
	}
	wall := time.Since(started)
	stats, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		return report{}, err
	}
	head := text.String()
	if len(head) > performanceTextHeadBytes {
		head = head[:performanceTextHeadBytes]
	}
	return report{
		ModelPath: modelPath, PromptTokens: promptTokens, GeneratedTokens: len(ids) - promptTokens,
		WallNS: uint64(wall.Nanoseconds()), PeakDeviceBytes: stats.PeakBytes,
		TextHead: head, FixtureSource: fixture.Source,
		GoVersion: runtime.Version(), OSArch: runtime.GOOS + "/" + runtime.GOARCH,
	}, nil
}

func writeReport(result report) error {
	payload, err := json.Marshal(result)
	if err != nil {
		return err
	}
	fmt.Println(string(payload))
	return nil
}

func runCheck(baselinePath, promptsPath string) error {
	var baseline sweepBaseline
	if err := jsonfile.Decode(baselinePath, &baseline); err != nil {
		return err
	}
	if err := validateSweepBaseline(baseline, promptsPath); err != nil {
		return err
	}
	var regressions []error
	for index, target := range baseline.Models {
		fmt.Printf("perf-sweep: checking %d/%d %s\n", index+1, len(baseline.Models), target.Name)
		current, err := measure(target.ModelPath, promptsPath)
		if err != nil {
			regressions = append(regressions, fmt.Errorf("%s: %w", target.Name, err))
			continue
		}
		if err := writeReport(current); err != nil {
			return err
		}
		if err := compareSweep(target, current, baseline); err != nil {
			regressions = append(regressions, fmt.Errorf("%s: %w", target.Name, err))
		}
	}
	if err := errors.Join(regressions...); err != nil {
		return err
	}
	fmt.Printf("perf-sweep: %d/%d installed baselines within wall and device-memory ceilings source_commit=%s\n",
		len(baseline.Models), len(baseline.Models), baseline.SourceCommit)
	fmt.Println("honesty: each model ran in this process against the fixed fixture; ceilings and source measurements are review-owned by the baseline")
	return nil
}

func validateSweepBaseline(baseline sweepBaseline, promptsPath string) error {
	if baseline.Schema != sweepBaselineSchema {
		return fmt.Errorf("baseline schema %q is not %q", baseline.Schema, sweepBaselineSchema)
	}
	if baseline.SourceCommit == "" {
		return errors.New("baseline carries no source commit")
	}
	if baseline.Fixture != promptsPath {
		return fmt.Errorf("baseline fixture %q differs from requested fixture %q", baseline.Fixture, promptsPath)
	}
	if !checked.NonzeroAll(baseline.RegressionScale, baseline.MaximumWallIncrease, baseline.MaximumPeakDeviceIncrease) ||
		!checked.Nonempty(baseline.Models) {
		return errors.New("baseline carries no regression policy or models")
	}
	seen := map[string]bool{}
	for _, target := range baseline.Models {
		if target.Name == "" || target.ModelPath == "" || !checked.PositiveInts(target.PromptTokens, target.GeneratedTokens) ||
			!checked.NonzeroAll(target.WallNS, target.PeakDeviceBytes) {
			return fmt.Errorf("baseline target %q is incomplete", target.Name)
		}
		if seen[target.Name] || seen[target.ModelPath] {
			return fmt.Errorf("baseline target %q is duplicated", target.Name)
		}
		seen[target.Name], seen[target.ModelPath] = true, true
	}
	return nil
}

func compareSweep(target sweepTarget, current report, baseline sweepBaseline) error {
	if current.PromptTokens != target.PromptTokens || current.GeneratedTokens != target.GeneratedTokens {
		return fmt.Errorf("token counts changed: prompt=%d/%d generated=%d/%d",
			current.PromptTokens, target.PromptTokens, current.GeneratedTokens, target.GeneratedTokens)
	}
	maximumWall, ok := regressionCeiling(target.WallNS, baseline.MaximumWallIncrease, baseline.RegressionScale)
	if !ok {
		return errors.New("wall regression ceiling overflows")
	}
	maximumPeak, ok := regressionCeiling(target.PeakDeviceBytes, baseline.MaximumPeakDeviceIncrease, baseline.RegressionScale)
	if !ok {
		return errors.New("device-memory regression ceiling overflows")
	}
	var regressions []error
	if current.WallNS > maximumWall {
		regressions = append(regressions, fmt.Errorf("wall_ns=%d exceeds ceiling=%d baseline=%d", current.WallNS, maximumWall, target.WallNS))
	}
	if current.PeakDeviceBytes > maximumPeak {
		regressions = append(regressions, fmt.Errorf("peak_device_bytes=%d exceeds ceiling=%d baseline=%d", current.PeakDeviceBytes, maximumPeak, target.PeakDeviceBytes))
	}
	return errors.Join(regressions...)
}

func regressionCeiling(measured, increase, scale uint64) (uint64, bool) {
	product, ok := checked.Mul64(measured, increase)
	if !ok || !checked.Nonzero(scale) {
		var zero uint64
		return zero, false
	}
	return checked.Add64(measured, product/scale)
}
