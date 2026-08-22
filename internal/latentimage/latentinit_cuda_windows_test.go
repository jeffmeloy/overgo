//go:build windows

package latentimage

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
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
	err = worker.Do(context.Background(), func(state *device.State) error {
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

// TestSeededInitLatentKreaGeometry exercises the real serving geometry: the
// [z_dim,256,256] latent for a 2048x2048 seed-42 Krea serve, with the shape
// DERIVED from the real checkpoint spec (VAE z_dim + spatial scale). It asserts
// the produced latent is exactly the golden init reduction -- bf16(torch.randn)
// of the same seeded draw -- over the full serving-size buffer, and every value
// is finite. Combined with the torch-anchored test above, the serving-shape
// latent is bit-exact to adaptive's mediaMixNoise init contract.
func TestSeededInitLatentKreaGeometry(t *testing.T) {
	cudatest.Require(t)
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	channels, height, width, err := media.DownsampledPlanarGeometry(spec.VAE.ZDim, 2048, 2048, spec.VAE.SpatialScale)
	if err != nil {
		t.Fatalf("LatentShape: %v", err)
	}
	shape := LatentShape{ZDim: channels, Height: height, Width: width}
	t.Logf("krea 2048x2048 latent geometry: [z=%d,h=%d,w=%d] elements=%d",
		shape.ZDim, shape.Height, shape.Width, shape.Elements())

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	latent := make([]float32, shape.Elements())
	reference := make([]float32, shape.Elements())
	err = worker.Do(context.Background(), func(state *device.State) error {
		got, err := SeededInitLatent(torchrng.NewStream(42), state, shape, InitNoiseMix)
		if err != nil {
			return err
		}
		copy(latent, got)
		// Independent draw of the SAME seeded stream, reduced by the pure host
		// core: the device-drawn latent must equal bf16(randn) element-for-element.
		ref := torchrng.NewStream(42)
		defer ref.Close(state)
		if err := ref.FillHost(state, reference); err != nil {
			return err
		}
		SeededLatentFromNoise(reference, InitNoiseMix)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	requireExactF32(t, latent, reference)
	for i, v := range latent {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite latent at %d: %v", i, v)
		}
	}
}
