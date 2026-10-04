//go:build windows

package inference

import (
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// sharedKVSteps decodes past the fixture's 4, 8, 16 and 32-token page bands.
const sharedKVSteps = 48

var sharedKVPrompt = []tokenizer.TokenID{3, 5, 8, 13}

func openSharedKVRunner(t *testing.T, path string, session modelrecipe.DecodeSessionPolicy) *Runner {
	t.Helper()
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		path, recipe.PlacementHybrid, session, recipe.ResidencyDeviceF32,
	)
	if err != nil {
		t.Fatalf("%s session: %v", session, err)
	}
	runner, err := OpenWithProgram(t.Context(), &loaded, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return runner
}

// generateSharedKV runs the raw-greedy device decode the server runs and
// reports the selected ids and the graphs the device replayed meanwhile.
func generateSharedKV(t *testing.T, path string, session modelrecipe.DecodeSessionPolicy) ([]tokenizer.TokenID, uint64) {
	t.Helper()
	runner := openSharedKVRunner(t, path, session)
	defer runner.Close()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	before, err := runner.DeviceExecutionStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
		MaxNewTokens: sharedKVSteps, Sampler: greedy, PromptTokenIDs: sharedKVPrompt,
		DeviceGreedy: true, ContinueAfterEOG: true,
	})
	if err != nil {
		t.Fatalf("%s session generate: %v", session, err)
	}
	after, err := runner.DeviceExecutionStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return ids, after.GraphLaunches - before.GraphLaunches
}

// stepSharedKVLogits prefills the prompt and feeds the argmax of each step's
// logits back for the same number of steps, keeping every step's logits.
func stepSharedKVLogits(t *testing.T, path string, session modelrecipe.DecodeSessionPolicy) [][]float32 {
	t.Helper()
	runner := openSharedKVRunner(t, path, session)
	defer runner.Close()
	runner.mu.Lock()
	defer runner.mu.Unlock()
	ctx := t.Context()
	var past *deviceKVCache
	defer func() {
		if past != nil {
			_ = past.Release(ctx)
		}
	}()
	tokens := slices.Clone(sharedKVPrompt)
	var steps [][]float32
	for range sharedKVSteps {
		next, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputLogits, tokens, past)
		if err != nil {
			t.Fatalf("%s session step %d: %v", session, len(steps), err)
		}
		if past != nil {
			_ = past.Release(ctx)
		}
		past = next
		steps = append(steps, slices.Clone(next.Logits))
		best := 0
		for index, logit := range next.Logits {
			if logit > next.Logits[best] {
				best = index
			}
		}
		tokens = []tokenizer.TokenID{tokenizer.TokenID(best)}
	}
	return steps
}

// TestSharedKVCapacitySessionDecodesLikeRequestSession decodes the miniature
// Gemma 4 E4B, whose last layers attend the KV of earlier ones, under the
// request session and under the capacity session across every page band, and
// demands the same greedy ids and the same logits bit for bit, and that only
// the capacity session replays graphs.
func TestSharedKVCapacitySessionDecodesLikeRequestSession(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := testutil.HermeticGemma4SharedKVGGUF(t, sharedKVFixtureContext)
	requestIDs, requestGraphs := generateSharedKV(t, path, modelrecipe.DecodeSessionRequest)
	capacityIDs, capacityGraphs := generateSharedKV(t, path, modelrecipe.DecodeSessionCapacity)
	if len(capacityIDs) != len(sharedKVPrompt)+sharedKVSteps || !slices.Equal(capacityIDs, requestIDs) {
		t.Fatalf("capacity ids %v differ from request ids %v", capacityIDs, requestIDs)
	}
	if requestGraphs != 0 || capacityGraphs == 0 {
		t.Fatalf("graph launches = request %d / capacity %d, want none and some", requestGraphs, capacityGraphs)
	}
	requestLogits := stepSharedKVLogits(t, path, modelrecipe.DecodeSessionRequest)
	capacityLogits := stepSharedKVLogits(t, path, modelrecipe.DecodeSessionCapacity)
	for step := range requestLogits {
		for index, want := range requestLogits[step] {
			if math.Float32bits(capacityLogits[step][index]) != math.Float32bits(want) {
				t.Fatalf("step %d logit %d = %g under capacity, %g under request",
					step, index, capacityLogits[step][index], want)
			}
		}
	}
}
