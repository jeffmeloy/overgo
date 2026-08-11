//go:build windows

package densecausal

import (
	"math/rand"
	"strconv"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// syntheticCausalModel builds a seeded dense-causal model at the requested
// geometry (tied embeddings, no attention bias) for timing/measurement.
func syntheticCausalModel(t *testing.T, seed int64, vocab, hidden, heads, headDim, kvHeads, inter, layers int) *Model {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	weights := map[string][]float32{}
	shapes := map[string][]int{}
	matrix := func(name string, rows, cols int) {
		values := make([]float32, rows*cols)
		for i := range values {
			values[i] = float32(rng.NormFloat64() * 0.02)
		}
		weights[name] = values
		shapes[name] = []int{rows, cols}
	}
	norm := func(name string, n int) {
		values := make([]float32, n)
		for i := range values {
			values[i] = float32(1 + rng.NormFloat64()*0.01)
		}
		weights[name] = values
		shapes[name] = []int{n}
	}
	matrix("model.embed_tokens.weight", vocab, hidden)
	for layer := 0; layer < layers; layer++ {
		prefix := "model.layers." + strconv.Itoa(layer) + "."
		norm(prefix+"input_layernorm.weight", hidden)
		matrix(prefix+"self_attn.q_proj.weight", heads*headDim, hidden)
		matrix(prefix+"self_attn.k_proj.weight", kvHeads*headDim, hidden)
		matrix(prefix+"self_attn.v_proj.weight", kvHeads*headDim, hidden)
		matrix(prefix+"self_attn.o_proj.weight", hidden, heads*headDim)
		norm(prefix+"post_attention_layernorm.weight", hidden)
		matrix(prefix+"mlp.gate_proj.weight", inter, hidden)
		matrix(prefix+"mlp.up_proj.weight", inter, hidden)
		matrix(prefix+"mlp.down_proj.weight", hidden, inter)
	}
	norm("model.norm.weight", hidden)
	m, err := NewModel(weights, shapes, heads, headDim, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestTrainingWallTimeHostVsGPU measures per-step wall time of host Train (host
// forward/backward + host Newton-Schulz Muon) vs TrainDeviceFull (host forward +
// device backward + device Muon) on a medium model. Reports the ratio; makes no
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
	tokens := make([]int, seq)
	rng := rand.New(rand.NewSource(7))
	for i := range tokens {
		tokens[i] = rng.Intn(vocab)
	}

	timeTrain := func(fn func(*Model) error) time.Duration {
		m := syntheticCausalModel(t, 1, vocab, hidden, heads, headDim, kvHeads, inter, layers)
		if err := fn(m); err != nil { // warm up (also JIT/context)
			t.Fatal(err)
		}
		m = syntheticCausalModel(t, 1, vocab, hidden, heads, headDim, kvHeads, inter, layers)
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
		_, e := m.TrainDeviceFull(worker, tokens, steps, 0, 0.9)
		return e
	})
	t.Logf("per-step: host Train %v, TrainDeviceFull %v (%.2fx)", hostT, devT, float64(hostT)/float64(devT))
}
