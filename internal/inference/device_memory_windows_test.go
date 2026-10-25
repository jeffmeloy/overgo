//go:build windows

package inference

import (
	"strings"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/modeldevice"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
)

// TestDeviceMemoryOwners: after a device generation the owners account for
// every live byte the runner holds, weights among them, within the device's
// driver-reported total.
func TestDeviceMemoryOwners(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openF32FixtureRunner(testutil.HermeticLlamaGGUF(t, hermeticContext), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := t.Context()
	if _, _, err := runner.Generate(ctx, "ab", GenerateOptions{MaxNewTokens: 4}); err != nil {
		t.Fatal(err)
	}
	memory, err := runner.DeviceMemory(ctx)
	if err != nil {
		t.Fatal(err)
	}
	stats, err := runner.DeviceMemoryStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var owned, weights uint64
	for _, owner := range memory.Owners {
		owned += owner.Bytes
		if owner.Name == "weights" {
			weights = owner.Bytes
		}
	}
	if owned != stats.CurrentBytes || weights == 0 || memory.TotalBytes == 0 || memory.FreeBytes > memory.TotalBytes || owned > memory.TotalBytes {
		t.Fatalf("owners %+v sum %d (weights %d) of live %d; device free %d of %d", memory.Owners, owned, weights, stats.CurrentBytes, memory.FreeBytes, memory.TotalBytes)
	}
	t.Logf("owners %+v; device free %d of %d", memory.Owners, memory.FreeBytes, memory.TotalBytes)
}

// TestAcceleratorTeardownReclaimsPromptLoans: the teardown the runner's closers share takes back a one-shot run's
// memory lent to a real expert residency while the residency lives, so it closes without an error.
func TestAcceleratorTeardownReclaimsPromptLoans(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	ctx := t.Context()
	file, err := gguf.Open(writeHermeticFlashNextGGUF(t))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var banks []gguf.TensorInfo
	for _, info := range file.Tensors {
		if strings.Contains(info.Name, "_exps.") {
			banks = append(banks, info)
		}
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	var weights *modeldevice.DeviceWeights
	var cuda *executor.Executor
	closed := false
	defer func() {
		if !closed {
			_ = closeAcceleratorResources(nil, weights, nil, cuda, worker)
		}
	}()
	if weights, err = modeldevice.NewDeviceWeights(worker); err != nil {
		t.Fatal(err)
	}
	loader := modeldevice.NewLoader(file)
	loader.Experts(weights, banks, 0)
	if err := loader.Load(ctx, nil); err != nil {
		t.Fatal(err)
	}
	if cuda, err = executor.NewWithWorker(worker); err != nil {
		t.Fatal(err)
	}
	if err := cuda.SetExpertBanks(weights); err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(64, 1))
	graph, err := executor.Compile(builder.Add(builder.Multiply(input, input), input))
	if err != nil {
		t.Fatal(err)
	}
	retained, err := cuda.ExecuteRetainedCompiledOnce(ctx, graph, map[*tensor.Tensor]reference.Value{
		input: {Shape: input.Shape, Data: make([]float32, 64)},
	}, nil, nil, nil)
	if err == nil {
		err = retained.Release(ctx)
	}
	if err == nil {
		err = cuda.ReleasePromptMemory(ctx)
	}
	if err != nil {
		t.Fatal(err)
	}
	if metrics, err := cuda.Metrics(ctx); err != nil || metrics.PromptArenaBytes == 0 || !weights.HasExpertResidency() {
		t.Fatalf("lent an arena of %d bytes to a residency %t (%v), want one lent", metrics.PromptArenaBytes, weights.HasExpertResidency(), err)
	}
	closed = true
	if err := closeAcceleratorResources(nil, weights, nil, cuda, worker); err != nil {
		t.Fatalf("closing after the loan: %v", err)
	}
}
