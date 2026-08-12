package latentimage

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

// torchRandnSeed31First is torch.manual_seed(31); torch.randn(17) on CUDA -- a
// REAL PyTorch bit-exact reference (the exact vector torchrng's acceptance oracle
// uses, ported from adaptive TestTorchCUDARandnCUDAStreamMatchesSequentialPyTorch
// Calls). Reproduced here to anchor the fill-order/NoiseMix logic to genuine
// torch.randn values with NO device: the model-free half of the latent-init proof.
var torchRandnSeed31First = []float32{
	-1.1408731, -0.20262821, -0.5782394, -0.6130378, -0.3019983, -0.18886788,
	-0.96660274, 1.6379809, 0.18491289, 0.5093818, -0.8460684, -0.6346741,
	0.739678, -0.7556086, -0.72963315, 1.8530686, 0.7509319,
}

func copyF32(src []float32) []float32 {
	out := make([]float32, len(src))
	copy(out, src)
	return out
}

func requireExactF32(t *testing.T, got, want []float32) {
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

// TestInitNoiseMixReducesToBF16Randn: the golden init contract (SignalWeight 0,
// NoiseWeight 1, NoiseScale 1, destination aliased to noise) applied to a real
// torch.randn draw yields bf16(randn) element-for-element, tol 0. This is the
// fill-order/NoiseWeight/NoiseScale reduction verified against genuine PyTorch
// values without CUDA -- the golden init latent is exactly RoundBF16(torch.randn).
func TestInitNoiseMixReducesToBF16Randn(t *testing.T) {
	if InitNoiseMix.SignalWeight != 0 || InitNoiseMix.NoiseWeight != 1 || InitNoiseMix.NoiseScale != 1 {
		t.Fatalf("InitNoiseMix drifted from the adaptive contract: %+v", InitNoiseMix)
	}
	got := SeededLatentFromNoise(copyF32(torchRandnSeed31First), InitNoiseMix)
	want := make([]float32, len(torchRandnSeed31First))
	for i, v := range torchRandnSeed31First {
		want[i] = dtype.RoundBF16(v)
	}
	requireExactF32(t, got, want)
}

// TestMixNoiseGeneralFormula: the general planar_mix_noise_bf16 formula matches a
// hand-computed reference (non-aliased dst != noise), covering SignalWeight != 0
// and NoiseScale != 1 (the refine/bridge paths the same primitive serves).
func TestMixNoiseGeneralFormula(t *testing.T) {
	m := NoiseMix{SignalWeight: 0.75, NoiseWeight: 0.25, NoiseScale: 1.5}
	noise := copyF32(torchRandnSeed31First)
	dst := make([]float32, len(noise))
	for i := range dst {
		dst[i] = float32(i)*0.1 - 0.8 // arbitrary distinct signal
	}
	want := make([]float32, len(noise))
	for i := range noise {
		n := dtype.RoundBF16(noise[i] * m.NoiseScale)
		want[i] = dtype.RoundBF16(m.SignalWeight*dst[i] + m.NoiseWeight*n)
	}
	mixNoiseInto(dst, noise, m)
	requireExactF32(t, dst, want)
}

// TestMixAliasedEqualsNonAliased: the in-place aliased call (the init contract)
// equals the non-aliased call when dst starts equal to noise -- proving the
// alias introduces no read-after-write hazard.
func TestMixAliasedEqualsNonAliased(t *testing.T) {
	m := NoiseMix{SignalWeight: 0.6, NoiseWeight: 0.4, NoiseScale: 1.0}
	aliased := copyF32(torchRandnSeed31First)
	mixNoiseInto(aliased, aliased, m)

	noise := copyF32(torchRandnSeed31First)
	dst := copyF32(torchRandnSeed31First) // dst == noise initially, distinct backing
	mixNoiseInto(dst, noise, m)

	requireExactF32(t, aliased, dst)
}

// TestSpecLatentShapeDerivation: latent geometry is config-derived (ZDim = VAE
// z_dim, spatial = pixel/SpatialScale) with no magic numbers, and rejects sizes
// not divisible by the VAE scale.
func TestSpecLatentShapeDerivation(t *testing.T) {
	// QwenImage VAE geometry: z_dim 16, dim_mult len 4 -> SpatialScale 2^3 = 8.
	spec := &Spec{VAE: VAESpec{ZDim: 16, SpatialScale: 8}}

	shape, err := spec.LatentShape(2048, 2048)
	if err != nil {
		t.Fatalf("LatentShape(2048,2048): %v", err)
	}
	if shape.ZDim != 16 || shape.Height != 256 || shape.Width != 256 {
		t.Fatalf("shape=%+v want {16,256,256}", shape)
	}
	if shape.Elements() != 16*256*256 {
		t.Fatalf("elements=%d want %d", shape.Elements(), 16*256*256)
	}

	if _, err := spec.LatentShape(2050, 2048); err == nil {
		t.Fatal("expected error for pixel size not divisible by vae scale")
	}
	if _, err := (&Spec{VAE: VAESpec{ZDim: 16, SpatialScale: 0}}).LatentShape(2048, 2048); err == nil {
		t.Fatal("expected error for zero vae scale")
	}
}
