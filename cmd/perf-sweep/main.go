// Command perf-sweep measures one servable model's inference against the
// fixed SFT prompt fixture: prompt context, generation wall, and peak device
// memory. One model per process so the device high-water belongs to exactly
// one artifact. The report is a single JSON line; recording it as a
// verification claim stays with the compatibility record producer.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

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
}

func main() {
	model := flag.String("model", "", "GGUF model path")
	prompts := flag.String("prompts", "fixtures/sweep_sft_prompts.json", "fixed SFT prompt fixture")
	flag.Parse()
	if *model == "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: perf-sweep -model <model.gguf> [-prompts fixture.json]")
		os.Exit(2)
	}
	if err := run(*model, *prompts); err != nil {
		fmt.Fprintln(os.Stderr, "perf-sweep:", err)
		os.Exit(1)
	}
}

func run(modelPath, promptsPath string) error {
	var fixture promptFixture
	if err := jsonfile.Decode(promptsPath, &fixture); err != nil {
		return err
	}
	if fixture.Prompt == "" || fixture.MaxTokens <= 0 {
		return fmt.Errorf("fixture %s carries no prompt or budget", promptsPath)
	}
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative,
	)
	if err != nil {
		return err
	}
	runner, err := inference.OpenWithProgram(context.Background(), &loaded, inference.OpenOptions{})
	if err != nil {
		return err
	}
	defer runner.Close()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		return err
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
		return err
	}
	wall := time.Since(started)
	stats, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		return err
	}
	head := text.String()
	if len(head) > 160 {
		head = head[:160]
	}
	payload, err := json.Marshal(report{
		ModelPath: modelPath, PromptTokens: promptTokens, GeneratedTokens: len(ids) - promptTokens,
		WallNS: uint64(wall.Nanoseconds()), PeakDeviceBytes: stats.PeakBytes,
		TextHead: head, FixtureSource: fixture.Source,
	})
	if err != nil {
		return err
	}
	fmt.Println(string(payload))
	return nil
}
