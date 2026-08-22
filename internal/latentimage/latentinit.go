// Seeded initial-latent construction for the Krea device serve. The geometry is
// DERIVED from the Spec (VAE z_dim + spatial scale), and the fill reproduces
// adaptive's mediaMixNoiseIntoDeviceCUDA init contract element-for-element: a
// torch.randn(seed) draw in channel-major contiguous order (torchrng, the
// PyTorch-bit-exact CUDA Philox source) mixed by planar_mix_noise_bf16. No magic
// numbers -- every dimension is config-read, and the bf16 mix is the ported
// kernel formula. The torchrng-backed draw lives in latentinit_cuda_windows.go
// (device-only); this file is the pure, CUDA-free core (geometry + mix).

package latentimage

import (
	"fmt"

	"overgo/internal/media"
)

// LatentShape is the [ZDim, Height, Width] geometry of a seeded initial latent,
// channel-major contiguous (matching a torch tensor of that shape and the flat
// fill order of adaptive's noise buffer). Height and Width are LATENT spatial
// dims: pixel dims divided by the VAE spatial downsample factor.
type LatentShape struct {
	ZDim   int // latent channels (VAE z_dim)
	Height int // latent rows (pixelHeight / SpatialScale)
	Width  int // latent cols (pixelWidth / SpatialScale)
}

// Elements is the flat element count ZDim*Height*Width (the torch.randn length).
func (s LatentShape) Elements() int { return s.ZDim * s.Height * s.Width }

func (s LatentShape) validate() error {
	if err := media.ValidatePlanarGeometry(s.ZDim, s.Height, s.Width); err != nil {
		return fmt.Errorf("latentimage: bad latent shape: %w", err)
	}
	return nil
}

// NoiseMix carries the scalars of adaptive's planar_mix_noise_bf16 kernel
// (go/extmodel/cuda_ops_cuda_windows.go). The destination is overwritten with
//
//	bf16( SignalWeight*dst + NoiseWeight*bf16(noise*NoiseScale) ).
type NoiseMix = media.NoiseMix

// InitNoiseMix is the seeded initial-latent contract for a fresh Krea serve:
// adaptive calls mediaMixNoiseIntoDeviceCUDA(Seed, NoiseWeight=1, NoiseScale=1)
// with the destination ALIASED to the noise buffer (joint_transformer_cuda_
// windows.go). SignalWeight is the zero value, so with the alias the mix reduces
// to bf16(randn): the raw torch.randn draw rounded once to bf16 storage.
var InitNoiseMix = media.FreshNoiseMix()

// mixNoiseInto reproduces planar_mix_noise_bf16 element-for-element in host f32.
// Round-to-nearest-even bf16 (dtype.RoundBF16) equals the device runtime_bf16 for
// every finite input, which every torch.randn draw is. dst and noise MAY alias
// (the init contract does): each element reads dst[i] and noise[i] before writing
// dst[i], so an in-place aliased call is exact. All arithmetic is float32,
// matching the kernel's __fmul_rn/__fadd_rn round-to-nearest-even.
func mixNoiseInto(dst, noise []float32, m NoiseMix) {
	media.MixNoiseBF16Into(dst, noise, m)
}

// SeededLatentFromNoise applies the NoiseMix contract to a pre-drawn noise buffer
// IN PLACE (dst aliased to noise, exactly as the adaptive init call site) and
// returns it. This is the pure, CUDA-free core of the seeded latent-init: given a
// torch.randn(seed) draw it yields the golden initial latent bit-for-bit.
func SeededLatentFromNoise(noise []float32, m NoiseMix) []float32 {
	mixNoiseInto(noise, noise, m)
	return noise
}
