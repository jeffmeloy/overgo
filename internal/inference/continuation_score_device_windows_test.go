//go:build windows

package inference

import (
	"math"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tokenizer"
)

// TestHermeticCUDAContinuationScoringParity: the device-cache scoring path
// and the host-cache path score the same prompt and candidates alike on
// the hermetic fixture, single-token and multi-token candidates included,
// within the logit tolerance the continuous-cache parity test holds.
// openHermeticScoringRunner opens the hermetic fixture on the device for
// the scoring tests and fails when it holds no resident device cache,
// the precondition of the device scoring path.
func openHermeticScoringRunner(t *testing.T) *Runner {
	t.Helper()
	requireIntegration(t)
	cudatest.Require(t)
	path := writeHermeticLlamaGGUFWithContext(t, hermeticContext)
	runner, err := openF32FixtureRunner(path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runner.Close() })
	if !runner.hasPreloadedWeights() || !runner.forwardProgram().PersistentDeviceCache() {
		t.Fatal("fixture runner does not hold a resident device cache")
	}
	return runner
}

func TestHermeticCUDAContinuationScoringParity(t *testing.T) {
	runner := openHermeticScoringRunner(t)
	prompt := []tokenizer.TokenID{1, 4, 5}
	continuations := [][]tokenizer.TokenID{{6}, {7, 4}, {5, 6, 7}, {2}}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	host, err := runner.scoreContinuationsHostLocked(t.Context(), prompt, continuations)
	if err != nil {
		t.Fatal(err)
	}
	device, err := runner.scoreContinuationsDeviceLocked(t.Context(), prompt, continuations)
	if err != nil {
		t.Fatal(err)
	}
	if len(host) != len(continuations) || len(device) != len(continuations) {
		t.Fatalf("score counts = %d/%d", len(host), len(device))
	}
	for index := range continuations {
		if host[index].Tokens != device[index].Tokens || host[index].Tokens != uint64(len(continuations[index])) {
			t.Fatalf("candidate %d token counts = %d/%d", index, host[index].Tokens, device[index].Tokens)
		}
		delta := math.Abs(host[index].LogProbability - device[index].LogProbability)
		t.Logf("candidate %d host=%.6f device=%.6f delta=%.2g", index, host[index].LogProbability, device[index].LogProbability, delta)
		if delta > 2e-4*float64(len(continuations[index])) {
			t.Fatalf("candidate %d log-probability delta %.3g", index, delta)
		}
	}
	// A second scoring of the same prompt must not read the first one's
	// candidate positions: the prompt cache is released and rebuilt.
	again, err := runner.scoreContinuationsDeviceLocked(t.Context(), prompt, continuations)
	if err != nil {
		t.Fatal(err)
	}
	for index := range continuations {
		if again[index] != device[index] {
			t.Fatalf("candidate %d rescored %+v, first %+v", index, again[index], device[index])
		}
	}
}
