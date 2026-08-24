//go:build windows && modeltest

package inference

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/checked"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
)

const (
	carbonWarmSamples                  = 11
	carbonMinimumGraphLaunchesPerChurn = 4
)

type carbonProcessGolden struct {
	Cases []struct {
		Prompt          string `json:"prompt"`
		MaxTokens       int    `json:"max_tokens"`
		Text            string `json:"text"`
		PromptTokens    int    `json:"prompt_tokens"`
		GeneratedTokens int    `json:"generated_tokens"`
	} `json:"cases"`
}

func TestCarbonServingProcessProbe(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	var golden carbonProcessGolden
	if err := jsonfile.Decode(testutil.FixturePath(t, "carbon_serving_golden.json"), &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("Carbon serving evidence has no cases")
	}
	testCase := golden.Cases[0]
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Carbon-500M-f16-ropefix.gguf")
	storePath := os.Getenv("OVERGO_CARBON_REPODB")
	if storePath == "" {
		t.Skip("child process probe requires OVERGO_CARBON_REPODB")
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	started := time.Now()
	loaded, err := modelrecipe.ResolveActiveGGUF(context.Background(), store, modelPath)
	if err != nil {
		t.Fatal(err)
	}
	resolvedAt := time.Now()
	runner, err := OpenWithProgram(context.Background(), &loaded, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	openedAt := time.Now()
	defer runner.Close()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	generate := func() time.Duration {
		promptTokens := 0
		requestStarted := time.Now()
		ids, fullText, generateErr := runner.Generate(context.Background(), testCase.Prompt, GenerateOptions{
			MaxNewTokens: testCase.MaxTokens,
			Sampler:      greedy,
			DeviceGreedy: true,
			OnPromptEvaluated: func(evaluation PromptEvaluation) {
				promptTokens = evaluation.Tokens
			},
		})
		if generateErr != nil {
			t.Fatal(generateErr)
		}
		generatedTokens := len(ids) - promptTokens
		text, ok := strings.CutPrefix(fullText, testCase.Prompt)
		if !ok {
			t.Fatalf("Carbon returned text %q lacks prompt %q", fullText, testCase.Prompt)
		}
		if promptTokens != testCase.PromptTokens || generatedTokens != testCase.GeneratedTokens || text != testCase.Text {
			t.Fatalf("Carbon output counts=%d/%d text=%q", promptTokens, generatedTokens, text)
		}
		return time.Since(requestStarted)
	}
	_ = generate()
	before, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	wall := time.Since(started)
	warm := make([]time.Duration, carbonWarmSamples)
	for index := range warm {
		warm[index] = generate()
	}
	slices.Sort(warm)
	memory, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	execution, err := runner.DeviceExecutionStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	instantiations := execution.GraphInstantiations - before.GraphInstantiations
	updates := execution.GraphUpdates - before.GraphUpdates
	graphLaunches := execution.GraphLaunches - before.GraphLaunches
	churn, churnOK := checked.Add64(instantiations, updates)
	scaledChurn, scaleOK := checked.Mul64(churn, carbonMinimumGraphLaunchesPerChurn)
	if !churnOK || !scaleOK || !checked.Nonzero(graphLaunches) || !checked.AtMost64(scaledChurn, graphLaunches) {
		t.Fatalf("Carbon decode graph replay is insufficient: churn=%d launches=%d",
			churn, graphLaunches)
	}
	fmt.Printf("CARBON_PROCESS_PROBE wall_ns=%d resolve_ns=%d open_ns=%d generate_ns=%d warm_ns=%d device_peak=%d graph_instantiations=%d graph_updates=%d graph_launches=%d kernel_launches=%d\n",
		wall.Nanoseconds(), resolvedAt.Sub(started).Nanoseconds(), openedAt.Sub(resolvedAt).Nanoseconds(),
		wall-openedAt.Sub(started), warm[len(warm)/2].Nanoseconds(), memory.PeakBytes,
		instantiations, updates, graphLaunches,
		execution.KernelLaunches-before.KernelLaunches)
}
