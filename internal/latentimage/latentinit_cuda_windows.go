//go:build windows

package latentimage

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/torchrng"
)

// SeededInitLatent draws the seeded initial latent for a Krea device serve. It
// fills shape.Elements() torch.randn(seed) normals from the torchrng stream (the
// PyTorch-bit-exact CUDA Philox source) in channel-major contiguous order --
// exactly the flat fill order of adaptive's mediaMixNoiseIntoDeviceCUDA over a
// [ZDim,H,W] tensor -- then applies the NoiseMix contract in place (destination
// aliased to the noise buffer, as the adaptive init call site does). With
// InitNoiseMix the result is bf16(randn) per element: the golden initial latent,
// ready to upload for the device denoiser. The stream's Philox counter advances,
// so a later fill on the same stream continues the same torch.randn sequence.
func SeededInitLatent(stream *torchrng.Stream, state *device.State, shape LatentShape, m NoiseMix) ([]float32, error) {
	if stream == nil {
		return nil, fmt.Errorf("latentimage: nil torchrng stream")
	}
	if err := shape.validate(); err != nil {
		return nil, err
	}
	noise := make([]float32, shape.Elements())
	if err := stream.FillHost(state, noise); err != nil {
		return nil, err
	}
	return SeededLatentFromNoise(noise, m), nil
}
