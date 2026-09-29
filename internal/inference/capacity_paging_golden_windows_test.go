//go:build windows

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/modeltest"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
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
// Gemma-4 append cache-write fix: Qwen3.5-4B on the capacity session must
// still match its READ-ONLY golden exactly.
func TestServingGoldenRegression(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	fixturePath := testutil.FixturePath(t, "qwen35_4b_serving_golden.json")
	var golden servingGolden
	if err := jsonfile.Decode(fixturePath, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) == 0 {
		t.Fatal("serving golden has no cases")
	}
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modeltest.Directory(t, "Qwen3.5-4B-f16.gguf"), recipe.PlacementHybrid,
		modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceF32,
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := OpenWithProgram(t.Context(), &loaded, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	for _, testCase := range golden.Cases {
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
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
