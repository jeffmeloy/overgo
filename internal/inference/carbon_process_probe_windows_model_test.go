//go:build windows && modeltest

package inference

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/repodb"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
)

func TestCarbonServingProcessProbe(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	var golden carbonServingGolden
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
		t.Fatal("OVERGO_CARBON_REPODB is required")
	}
	store, err := repodb.Open(storePath)
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
	var text strings.Builder
	promptTokens := 0
	ids, _, err := runner.Generate(context.Background(), testCase.Prompt, GenerateOptions{
		MaxNewTokens: testCase.MaxTokens,
		Sampler:      greedy,
		DeviceGreedy: true,
		OnToken: func(event TokenEvent) error {
			text.WriteString(event.Piece)
			return nil
		},
		OnPromptEvaluated: func(evaluation PromptEvaluation) {
			promptTokens = evaluation.Tokens
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	wall := time.Since(started)
	memory, err := runner.DeviceMemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	generatedTokens := len(ids) - promptTokens
	if promptTokens != testCase.PromptTokens || generatedTokens != testCase.GeneratedTokens || text.String() != testCase.Text {
		t.Fatalf("Carbon output counts=%d/%d text=%q", promptTokens, generatedTokens, text.String())
	}
	fmt.Printf("CARBON_PROCESS_PROBE wall_ns=%d resolve_ns=%d open_ns=%d generate_ns=%d device_peak=%d\n",
		wall.Nanoseconds(), resolvedAt.Sub(started).Nanoseconds(), openedAt.Sub(resolvedAt).Nanoseconds(),
		wall-openedAt.Sub(started), memory.PeakBytes)
}
