//go:build windows

package inference

import (
	"context"
	"os"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/tokenizer"
)

func generateCapacityPagingRepro(
	t *testing.T, path string, session modelrecipe.DecodeSessionPolicy,
) []tokenizer.TokenID {
	t.Helper()
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		path, recipe.PlacementHybrid, session, recipe.ResidencyDeviceF32,
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := OpenWithProgram(&loaded, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if !greedy.IsRawGreedy() {
		t.Fatal("sampler is not raw greedy")
	}
	ids, _, err := runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   300,
		Sampler:        greedy,
		PromptTokenIDs: []tokenizer.TokenID{1, 2, 3, 4},
		DeviceGreedy:   true,
	})
	if err != nil {
		t.Fatalf("%s-session generate across page boundary failed: %v", session, err)
	}
	return ids
}

// TestCapacityPagingReproGeneratesPastPageBoundary drives the real serve path
// (raw-greedy device decode on the default capacity session) past the 256-token
// KV-cache page boundary on a capacity-eligible model, and asserts the output is
// token-identical to the proven-correct request (concat) session. Set
// OVERGO_REPRO_MODEL to a capacity-eligible GGUF (Carbon/MiniCPM/Qwen35-4B, or the
// gemma-4-12B). Before the fix the capacity session OOMed at pastTokens=256.
func TestCapacityPagingReproGeneratesPastPageBoundary(t *testing.T) {
	cudatest.Require(t)
	path := os.Getenv("OVERGO_REPRO_MODEL")
	if path == "" {
		t.Skip("OVERGO_REPRO_MODEL is not set")
	}
	capacityIDs := generateCapacityPagingRepro(t, path, modelrecipe.DecodeSessionCapacity)
	if len(capacityIDs) <= 256 {
		t.Fatalf("capacity session generated %d ids, need >256 to cross the page boundary", len(capacityIDs))
	}
	requestIDs := generateCapacityPagingRepro(t, path, modelrecipe.DecodeSessionRequest)
	if !slices.Equal(capacityIDs, requestIDs) {
		t.Fatalf("capacity vs request ids differ: capacity=%d request=%d", len(capacityIDs), len(requestIDs))
	}
	t.Logf("capacity == request: %d ids, crossed the 256 boundary token-identically", len(capacityIDs))
}
