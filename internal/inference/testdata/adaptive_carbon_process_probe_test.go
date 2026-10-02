//go:build cuda && windows

// This file is the reference child of overgo's Carbon cold-lifecycle
// comparison (TestCarbonColdLifecyclePhases in overgo's internal/inference).
// Overgo copies it into adaptive_new's go/cmd/adaptive-gpt-server at commit
// 214950b, builds that package with -tags cuda and runs this one test as an
// isolated process, alternating with overgo's own TestCarbonServingProcessProbe.
//
// It performs on adaptive's serving engine the lifecycle the candidate performs
// on overgo's, for the first case of the same golden file. Adaptive serves
// Carbon from its models/Carbon-500M directory, the checkpoint overgo's GGUF is
// converted from (214950b refuses the converted GGUF: architecture "llama"
// declares no serving identity). The parent seeds that directory's active
// inference recipe with adaptive's own seed-model-profiles into a fresh store
// and redirects adaptive's catalog to it, as adaptive's commit gate does.
//
//   - resolve_ns: engineForRoot binds the model to its active inference recipe;
//   - open_ns: the engine's Load makes the model resident on the device;
//   - generate_ns: the first greedy raw-prompt request;
//   - wall_ns: resolve, open and the first request together, the cold lifecycle;
//   - warm_ns: the median of 11 further identical requests;
//   - device_peak: the peak of adaptive's Go-owned CUDA allocations
//     (matrix.StartCUDAOwnedAllocationObserver) across all of the above.
//
// Every request must reproduce the golden prompt and generated token counts, so
// both sides do the same work; text_match reports whether adaptive's greedy
// text also equals the golden, which overgo generated. The candidate's graph
// counters have no adaptive counterpart and are not printed.
//
// Acceptance: the probe behind overgo's recorded adaptive figures was an
// uncommitted file that is lost. This replacement is accepted when it
// reproduces them within run-to-run noise: 1.689 s lifecycle, 3,798.583 MiB
// owned device and 3,884.645 MiB host peak (overgo 21ecd5ff1,
// docs/adaptive_new_parity_report.md:106), and 68.796 ms warm generation
// (overgo 8084621b2, same row). Those runs used the golden's earlier first case,
// an 8-token prompt with the same 32 generated tokens. Reproduced 2026-10-02:
// 1.649 s lifecycle, 3,798.583 MiB owned device, about 3,898 MiB host and
// 68.9 ms warm generation.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/jeffmeloy/adaptive_gpt/go/matrix"
)

const carbonProcessProbeWarmSamples = 11

func TestCarbonServingProcessProbe(t *testing.T) {
	modelDir, goldenPath := os.Getenv("OVERGO_CARBON_ADAPTIVE_MODEL"), os.Getenv("OVERGO_CARBON_GOLDEN")
	// Only overgo's comparison runs this file, and it always sets both.
	if modelDir == "" || goldenPath == "" {
		t.Fatal("reference probe requires OVERGO_CARBON_ADAPTIVE_MODEL and OVERGO_CARBON_GOLDEN")
	}
	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Cases []struct {
			Prompt          string `json:"prompt"`
			MaxTokens       int    `json:"max_tokens"`
			Text            string `json:"text"`
			PromptTokens    int    `json:"prompt_tokens"`
			GeneratedTokens int    `json:"generated_tokens"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("Carbon serving evidence has no cases")
	}
	testCase := golden.Cases[0]
	observer := matrix.StartCUDAOwnedAllocationObserver()
	started := time.Now()
	// The serving root is the directory holding the model, where the parent's
	// seed named it.
	engine, admitted, err := engineForRoot(filepath.Dir(modelDir), modelDir)
	if err != nil {
		t.Fatal(err)
	}
	resolvedAt := time.Now()
	defer engine.Close()
	if err := engine.Load(admitted); err != nil {
		t.Fatal(err)
	}
	openedAt := time.Now()
	textMatch := true
	generate := func() time.Duration {
		requestStarted := time.Now()
		text, generatedTokens, _, extras, generateErr := engine.Generate(context.Background(), testCase.Prompt, testCase.MaxTokens, 0, false)
		if generateErr != nil {
			t.Fatal(generateErr)
		}
		promptTokens, _ := extras["prompt_tokens"].(int)
		if promptTokens != testCase.PromptTokens || generatedTokens != testCase.GeneratedTokens {
			t.Fatalf("Carbon output counts=%d/%d text=%q", promptTokens, generatedTokens, text)
		}
		textMatch = textMatch && text == testCase.Text
		return time.Since(requestStarted)
	}
	_ = generate()
	wall := time.Since(started)
	warm := make([]time.Duration, carbonProcessProbeWarmSamples)
	for index := range warm {
		warm[index] = generate()
	}
	slices.Sort(warm)
	memory := observer.Stop()
	fmt.Printf("CARBON_PROCESS_PROBE wall_ns=%d resolve_ns=%d open_ns=%d generate_ns=%d warm_ns=%d device_peak=%d text_match=%t\n",
		wall.Nanoseconds(), resolvedAt.Sub(started).Nanoseconds(), openedAt.Sub(resolvedAt).Nanoseconds(),
		(wall - openedAt.Sub(started)).Nanoseconds(), warm[len(warm)/2].Nanoseconds(), memory.PeakOwnedBytes, textMatch)
}
