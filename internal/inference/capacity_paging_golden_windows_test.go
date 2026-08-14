//go:build windows

package inference

import (
	"context"
	"os"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/tokenizer"
)

type servingGoldenCase struct {
	Name         string              `json:"name"`
	PromptIDs    []tokenizer.TokenID `json:"prompt_ids"`
	GeneratedIDs []tokenizer.TokenID `json:"generated_ids"`
}

type servingGolden struct {
	Cases []servingGoldenCase `json:"cases"`
}

// TestServingGoldenRegression re-serves a serving-golden fixture through the real
// device path and asserts token-for-token equality. Regression guard for the
// Gemma-4 append cache-write fix: E4B (request session, SharedKV) and Qwen3.5-4B
// (capacity session) must still match their READ-ONLY goldens exactly.
//
// Env: OVERGO_GOLDEN_MODEL (gguf), OVERGO_GOLDEN_FIXTURE (json),
// OVERGO_GOLDEN_SESSION (request|capacity).
func TestServingGoldenRegression(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	modelPath := os.Getenv("OVERGO_GOLDEN_MODEL")
	fixturePath := os.Getenv("OVERGO_GOLDEN_FIXTURE")
	sessionName := os.Getenv("OVERGO_GOLDEN_SESSION")
	if modelPath == "" || fixturePath == "" || sessionName == "" {
		t.Skip("set OVERGO_GOLDEN_MODEL, OVERGO_GOLDEN_FIXTURE, OVERGO_GOLDEN_SESSION")
	}
	var session modelrecipe.DecodeSessionPolicy
	switch sessionName {
	case "request":
		session = modelrecipe.DecodeSessionRequest
	case "capacity":
		session = modelrecipe.DecodeSessionCapacity
	default:
		t.Fatalf("OVERGO_GOLDEN_SESSION = %q, want request|capacity", sessionName)
	}
	var golden servingGolden
	if err := jsonfile.Decode(fixturePath, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("serving golden has no cases")
	}
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, session, recipe.ResidencyDeviceF32,
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := OpenWithProgram(&loaded, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	for _, testCase := range golden.Cases {
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		ids, _, err := runner.Generate(context.Background(), "", GenerateOptions{
			MaxNewTokens:   len(testCase.GeneratedIDs),
			Sampler:        greedy,
			PromptTokenIDs: testCase.PromptIDs,
			DeviceGreedy:   true,
		})
		if err != nil {
			t.Fatalf("case %q generate failed: %v", testCase.Name, err)
		}
		got := ids[len(ids)-len(testCase.GeneratedIDs):]
		if !slices.Equal(got, testCase.GeneratedIDs) {
			t.Fatalf("case %q generated %v, want %v", testCase.Name, got, testCase.GeneratedIDs)
		}
	}
	t.Logf("serving golden %s: %d cases token-identical", fixturePath, len(golden.Cases))
}
