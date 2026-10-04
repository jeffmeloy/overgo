//go:build windows

package latentimage

import (
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
)

// torchRandnSeed31Second is torch.manual_seed(31); the SECOND torch.randn(13)
// continuing the same Philox stream (ported from the adaptive/torchrng oracle).
// A single stream draws First then Second as one contiguous torch.randn sequence.
var torchRandnSeed31Second = []float32{
	1.2934998, -1.1376057, -0.48691002, -1.8742987, 1.5954318, -0.454027,
	2.2399902, 0.07513574, -0.033937573, -0.59114724, 0.75303495, -1.2242086,
	0.67537993,
}

// TestSeededInitLatentMatchesTorchRandn31 is the tol-0 bit-exact anchor of the
// full init contract to REAL PyTorch: SeededInitLatent(seed 31) with the golden
// InitNoiseMix reproduces bf16(torch.randn(31)) for two successive draws that
// continue the same Philox stream. This proves the torchrng draw + fill-order +
// NoiseWeight/NoiseScale reduction end-to-end on the 4090D, tol 0.
func TestSeededInitLatentMatchesTorchRandn31(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	first := make([]float32, len(torchRandnSeed31First))
	second := make([]float32, len(torchRandnSeed31Second))
	err = worker.Do(t.Context(), func(state *device.State) error {
		stream := torchrng.NewStream(31)
		defer stream.Close(state)
		// 17 and 13 elements as [N,1,1] latent shapes: the flat torch.randn length
		// is what the fill contract fixes; the CHW factoring is exercised separately.
		got, err := SeededInitLatent(stream, state, LatentShape{ZDim: len(first), Height: 1, Width: 1}, InitNoiseMix)
		if err != nil {
			return err
		}
		copy(first, got)
		got, err = SeededInitLatent(stream, state, LatentShape{ZDim: len(second), Height: 1, Width: 1}, InitNoiseMix)
		if err != nil {
			return err
		}
		copy(second, got)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	wantFirst := make([]float32, len(torchRandnSeed31First))
	for i, v := range torchRandnSeed31First {
		wantFirst[i] = dtype.RoundBF16(v)
	}
	wantSecond := make([]float32, len(torchRandnSeed31Second))
	for i, v := range torchRandnSeed31Second {
		wantSecond[i] = dtype.RoundBF16(v)
	}
	requireExactF32(t, first, wantFirst)
	requireExactF32(t, second, wantSecond)
}
