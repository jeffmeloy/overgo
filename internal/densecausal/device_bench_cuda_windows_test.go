//go:build windows

package densecausal

import (
	"math/rand"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

func syntheticCausalModel(t *testing.T, spec testutil.DenseCausalSpec) *Model {
	t.Helper()
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	m, err := NewModel(weights, shapes, spec.Heads, spec.HeadDim, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestTrainingWallTimeHostVsGPU measures per-step wall time of host Train (host
// forward/backward + host Newton-Schulz Muon) vs resident device training on a
// medium model. Reports the ratio; makes no
// speed assertion (informational, the rung's measurement). Named without "Device"
// so the gate's `-run Device` device lane does NOT run this ~8-minute benchmark;
// run it manually with -run TestTrainingWallTime.
func TestTrainingWallTimeHostVsGPU(t *testing.T) {
	if testing.Short() {
		t.Skip("timing benchmark; skipped in -short")
	}
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	// Medium geometry where host Newton-Schulz on the big matrices is a real cost.
	const (
		vocab, hidden, heads, headDim, kvHeads, inter, layers = 256, 512, 8, 64, 2, 1536, 2
		seq                                                   = 24
		steps                                                 = 3
	)
	modelSpec := testutil.DenseCausalSpec{
		Vocab: vocab, Hidden: hidden, Heads: heads, HeadDim: headDim,
		KVHeads: kvHeads, Intermediate: inter, Layers: layers, Seed: 1,
	}
	tokens := make([]int, seq)
	rng := rand.New(rand.NewSource(7))
	for i := range tokens {
		tokens[i] = rng.Intn(vocab)
	}

	timeTrain := func(fn func(*Model) error) time.Duration {
		m := syntheticCausalModel(t, modelSpec)
		if err := fn(m); err != nil { // warm up (also JIT/context)
			t.Fatal(err)
		}
		m = syntheticCausalModel(t, modelSpec)
		start := time.Now()
		if err := fn(m); err != nil {
			t.Fatal(err)
		}
		return time.Since(start) / steps
	}

	hostT := timeTrain(func(m *Model) error {
		_, e := m.Train(tokens, steps, 0, 0.9)
		return e
	})
	devT := timeTrain(func(m *Model) error {
		_, e := m.TrainDeviceResident(worker, tokens, steps, 0, 0.9)
		return e
	})
	t.Logf("per-step: host Train %v, resident device %v (%.2fx)", hostT, devT, float64(hostT)/float64(devT))
}
