//go:build windows

package inference

import (
	"math"
	"os"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
)

// TestContinuationScoringPathsAgreeOnAModel: the device-cache scoring
// path and the host-cache path agree on a real model, at native
// precision, on a long prompt with multi-token candidates and on a long
// single sequence scored from the model's scoring prefix (the DNA
// suite's shape). OVERGO_PARITY_MODEL names the GGUF; unset, the test
// is skipped. The hermetic fixture test holds the same parity on a toy
// model; this one holds it where the numbers are recorded.
// scoringParityRequest is one scoring request the parity tests put to
// two runners or two paths.
type scoringParityRequest struct {
	name       string
	prompt     string
	candidates []string
}

// scoringParityRequests picks the requests a model's domain makes
// meaningful: a model with a scoring prefix (a DNA tokenizer) scores a
// sequence from it, where English text is out of its domain and its
// half-precision noise says nothing; a text model scores a long prompt
// with letter and phrase candidates.
func scoringParityRequests(runner *Runner, sentences int) []scoringParityRequest {
	if prefix := runner.ScoringPrefix(); prefix != "" {
		return []scoringParityRequest{
			{"sequence from the scoring prefix", prefix, []string{strings.Repeat("ACGTTGCA", 40)}},
		}
	}
	sentence := "The committee met on Tuesday to review the harbor survey and the bridge tender's log. "
	prompt := strings.Repeat(sentence, sentences) + "\nAnswer:"
	return []scoringParityRequest{
		{"long prompt, letters", prompt, []string{" A", " B", " C"}},
		{"long prompt, phrases", prompt, []string{" the kitchen counter", " under the guest bed", " in the hall closet"}},
	}
}

// openParityRunner opens a local model under a residency, taking the
// decode session policy the model's plan accepts.
func openParityRunner(t *testing.T, path string, residency recipe.ResidencyPolicy) *Runner {
	t.Helper()
	var lastErr error
	for _, session := range []modelrecipe.DecodeSessionPolicy{modelrecipe.DecodeSessionCapacity, modelrecipe.DecodeSessionRequest} {
		loaded, err := servingtest.ResolveActiveGGUFWithPolicy(path, recipe.PlacementHybrid, session, residency)
		if err != nil {
			lastErr = err
			continue
		}
		runner, err := OpenWithProgram(t.Context(), &loaded, OpenOptions{})
		if err != nil {
			lastErr = err
			continue
		}
		t.Cleanup(func() { _ = runner.Close() })
		return runner
	}
	t.Fatal(lastErr)
	return nil
}

func TestContinuationScoringMatchesReferenceBackend(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := os.Getenv("OVERGO_PARITY_MODEL")
	if path == "" {
		t.Skip("set OVERGO_PARITY_MODEL to a GGUF path")
	}
	// The host-reference residency runs every operator on the CPU
	// reference backend; the device runner scores through its kernels.
	// The two must agree on a long single sequence from the model's
	// scoring prefix (the DNA suite's shape) and on a choice prompt.
	reference := openParityRunner(t, path, recipe.ResidencyHostReference)
	device := openParityRunner(t, path, recipe.ResidencyHybridNative)
	// The reference backend runs the model on the CPU, so its prompt is
	// short; the device paths' test takes the long one.
	for _, request := range scoringParityRequests(device, 4) {
		t.Run(request.name, func(t *testing.T) {
			want, err := reference.ScoreContinuations(t.Context(), request.prompt, request.candidates)
			if err != nil {
				t.Fatal(err)
			}
			got, err := device.ScoreContinuations(t.Context(), request.prompt, request.candidates)
			if err != nil {
				t.Fatal(err)
			}
			for index := range request.candidates {
				delta := math.Abs(want[index].LogProbability - got[index].LogProbability)
				t.Logf("candidate %d tokens=%d reference=%.4f device=%.4f delta=%.3g", index, want[index].Tokens, want[index].LogProbability, got[index].LogProbability, delta)
				if want[index].Tokens != got[index].Tokens || delta > 5e-2*float64(want[index].Tokens) {
					t.Fatalf("candidate %d: reference %+v, device %+v", index, want[index], got[index])
				}
			}
		})
	}
}

func TestContinuationScoringPathsAgreeOnAModel(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := os.Getenv("OVERGO_PARITY_MODEL")
	if path == "" {
		t.Skip("set OVERGO_PARITY_MODEL to a GGUF path")
	}
	runner := openParityRunner(t, path, recipe.ResidencyHybridNative)
	if !runner.hasPreloadedWeights() || !runner.forwardProgram().PersistentDeviceCache() {
		t.Skip("model holds no resident device cache")
	}
	for _, request := range scoringParityRequests(runner, 60) {
		t.Run(request.name, func(t *testing.T) {
			promptIDs, continuations, err := runner.continuationTokens(request.prompt, request.candidates, false)
			if err != nil {
				t.Fatal(err)
			}
			runner.mu.Lock()
			defer runner.mu.Unlock()
			host, err := runner.scoreContinuationsHostLocked(t.Context(), promptIDs, continuations)
			if err != nil {
				t.Fatal(err)
			}
			device, err := runner.scoreContinuationsDeviceLocked(t.Context(), promptIDs, continuations)
			if err != nil {
				t.Fatal(err)
			}
			for index := range continuations {
				delta := math.Abs(host[index].LogProbability - device[index].LogProbability)
				t.Logf("candidate %d tokens=%d host=%.4f device=%.4f delta=%.3g", index, host[index].Tokens, host[index].LogProbability, device[index].LogProbability, delta)
				// The host path prefills a longer candidate through the
				// tensor-core GEMM while the device path decodes it a token at
				// a time through the F32 kernels; at native precision they
				// differ by a few hundredths of a nat per token, and a wrong
				// alignment would differ by whole nats.
				if host[index].Tokens != device[index].Tokens || delta > 5e-2*float64(host[index].Tokens) {
					t.Fatalf("candidate %d: host %+v, device %+v", index, host[index], device[index])
				}
			}
		})
	}
}
