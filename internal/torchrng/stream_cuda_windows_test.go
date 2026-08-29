//go:build windows

package torchrng

import (
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
)

// PyTorch golden values: torch.manual_seed(31) on CUDA, two successive
// randn() calls of length 17 then 13 (the second continues the same Philox
// stream). Ported verbatim from adaptive_new
// TestTorchCUDARandnCUDAStreamMatchesSequentialPyTorchCalls. Bit-exact (tol 0).
var (
	goldenSeed31First = []float32{
		-1.1408731, -0.20262821, -0.5782394, -0.6130378, -0.3019983, -0.18886788,
		-0.96660274, 1.6379809, 0.18491289, 0.5093818, -0.8460684, -0.6346741,
		0.739678, -0.7556086, -0.72963315, 1.8530686, 0.7509319,
	}
	goldenSeed31Second = []float32{
		1.2934998, -1.1376057, -0.48691002, -1.8742987, 1.5954318, -0.454027,
		2.2399902, 0.07513574, -0.033937573, -0.59114724, 0.75303495, -1.2242086,
		0.67537993,
	}
)

func requireExact(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mismatch at %d: got=%v want=%v (bit-exact required)", i, got[i], want[i])
		}
	}
}

// TestTorchRNGStreamMatchesSequentialPyTorch is the acceptance oracle: a single
// Stream(31) reproduces two successive torch.randn calls bit-for-bit, and its
// Philox offset advances between them.
func TestTorchRNGStreamMatchesSequentialPyTorch(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	first := make([]float32, len(goldenSeed31First))
	second := make([]float32, len(goldenSeed31Second))
	var firstOffset, finalOffset uint64
	err = worker.Do(t.Context(), func(state *device.State) error {
		stream := NewStream(31)
		defer stream.Close(state)
		if err := stream.FillHost(state, first); err != nil {
			return err
		}
		firstOffset = stream.Offset()
		if err := stream.FillHost(state, second); err != nil {
			return err
		}
		finalOffset = stream.Offset()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	requireExact(t, first, goldenSeed31First)
	requireExact(t, second, goldenSeed31Second)
	if firstOffset == 0 || finalOffset <= firstOffset {
		t.Fatalf("offsets first=%d final=%d (must advance)", firstOffset, finalOffset)
	}
}

// TestTorchRNGDeviceMatchesHost proves the FillDevice path equals the FillHost
// path bit-for-bit for identically seeded streams, across two successive fills,
// with matching final offsets. This is the neutral device==host parity check
// (the adaptive channel-major test carried a video-specific transpose that does
// not belong in a family-neutral RNG home).
func TestTorchRNGDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const first, second = 40, 24
	host := make([]float32, first+second)
	fromDevice := make([]float32, first+second)
	var hostOffset, deviceOffset uint64
	err = worker.Do(t.Context(), func(state *device.State) error {
		hostStream := NewStream(31)
		defer hostStream.Close(state)
		if err := hostStream.FillHost(state, host[:first]); err != nil {
			return err
		}
		if err := hostStream.FillHost(state, host[first:]); err != nil {
			return err
		}
		hostOffset = hostStream.Offset()

		deviceStream := NewStream(31)
		defer deviceStream.Close(state)
		scratch, err := state.Driver.MemAlloc(uint64(first+second) * 4)
		if err != nil {
			return err
		}
		defer func() { _ = state.Driver.MemFree(scratch) }()
		if err := deviceStream.FillDevice(state, scratch, first); err != nil {
			return err
		}
		if err := deviceStream.FillDevice(state, scratch+driver.DevicePtr(first*4), second); err != nil {
			return err
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		if err := state.Driver.MemcpyDtoH(driver.Bytes(fromDevice), scratch); err != nil {
			return err
		}
		deviceOffset = deviceStream.Offset()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	requireExact(t, fromDevice, host)
	if deviceOffset != hostOffset {
		t.Fatalf("device offset=%d host offset=%d (must match)", deviceOffset, hostOffset)
	}
}
