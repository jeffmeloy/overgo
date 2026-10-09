//go:build windows

package latentvideo

import (
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
)

// clipCodecCeiling is the time the autoencoder may take to encode a clip of 21
// frames at 832 by 480, and the time it may take to decode one. Encoding took
// 8.1 seconds and decoding 10.1 before the codec's convolutions were computed
// in tiles of registers, and they take 3.4 and 3.3; the ceiling stands between,
// so the earlier kernels coming back fail here. The autoencoder's own code, on
// the same device over a clip of the same size, encodes in 0.7 to 1.0 seconds
// and decodes in 1.25 (scripts/python_golden_references/liveedit_own_edit.py):
// it computes in bfloat16 on the device's tensor units, where this codec keeps
// the float32 sums that hold it to its host reference value for value.
const clipCodecCeiling = 5 * time.Second

// TestClipCodecKeepsToItsOwnCodesTime times the autoencoder over a clip of the
// size a served edit carries, 21 frames at 832 by 480: the source clip encoded
// to its latent, and that latent decoded to a clip.
func TestClipCodecKeepsToItsOwnCodesTime(t *testing.T) {
	cudatest.Require(t)
	directory := wanModelDir(t)
	checkpoint := AutoencoderWeights(directory)
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		t.Skipf("UNAVAILABLE: %v", err)
	}
	profile, err := ResolveProfile(directory)
	if err != nil {
		t.Fatal(err)
	}
	encoderGraph, err := CompileVAEEncoderPlan(catalog.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	decoderPlan, err := CompileVAEDecoderPlan(catalog.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	const frames, height, width = 21, 480, 832
	plan, err := CompileSourceCodecBoundary(SourceCodecProfile{
		InputChannels: encoderGraph.InputChannels, LatentChannels: encoderGraph.LatentChannels,
		Stride: encoderGraph.Stride, LatentStats: profile.LatentStats,
	}, SourceVideoShape{Channels: media.RGBChannels, Frames: frames, Height: height, Width: width})
	if err != nil {
		t.Fatal(err)
	}
	plane := height * width
	source := make([]float32, media.RGBChannels*frames*plane)
	for channel := range media.RGBChannels {
		for frame := range frames {
			for at := range plane {
				x, y := at%width, at/width
				source[(channel*frames+frame)*plane+at] = float32((x+2*y+7*frame+40*channel)%width) / float32(width)
			}
		}
	}
	encoder, err := newVAEEncoderCUDASession(checkpoint, encoderGraph, device.DefaultOrdinal())
	if err != nil {
		t.Fatal(err)
	}
	defer encoder.Close()
	started := time.Now()
	latent, _, err := encoder.Encode(t.Context(), plan, source)
	if err != nil {
		t.Fatal(err)
	}
	encoded := time.Since(started)
	decoder, err := NewVAEDecoderCUDASession(checkpoint, decoderPlan, device.DefaultOrdinal())
	if err != nil {
		t.Fatal(err)
	}
	defer decoder.Close()
	started = time.Now()
	stats, err := decoder.DecodeContext(t.Context(), profile.LatentStats, latent, plan.Latent.Frames, plan.Latent.Height, plan.Latent.Width,
		func(int, []float32, int, int) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	decoded := time.Since(started)
	if stats.OutputFrames != frames {
		t.Fatalf("decoded %d frames of %d", stats.OutputFrames, frames)
	}
	t.Logf("a clip of %d frames at %d by %d: encoded in %.2f s, decoded in %.2f s", frames, width, height, encoded.Seconds(), decoded.Seconds())
	if encoded > clipCodecCeiling || decoded > clipCodecCeiling {
		t.Errorf("the clip took %.2f s to encode and %.2f s to decode, where each is held to %v", encoded.Seconds(), decoded.Seconds(), clipCodecCeiling)
	}
}
