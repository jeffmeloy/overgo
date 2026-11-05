//go:build windows && modeltest

package mosstrain

import (
	"context"
	"math/rand/v2"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestMOSSAdapterStepOnTheCheckpoint measures one adapter step of the real
// transformer: its wall, its peak device memory and whether its loss falls on
// the example it stepped on.
func TestMOSSAdapterStepOnTheCheckpoint(t *testing.T) {
	cudatest.Require(t)
	root := testutil.ModelArtifactDir(t, "MOSS-Image-SFT-797fa8e678b4f")
	spec, err := latentimage.ReadRotaryTransformer(root)
	testcheck.NoError(t, err)
	source, err := safetensors.OpenSource(filepath.Join(root, "transformer"))
	testcheck.NoError(t, err)
	defer source.Close()
	options := Options{Rank: 16, Alpha: 16}
	config := optimizer.Config{BaseLearningRate: 1e-3, Momentum: 0.9, Schedule: optimizer.ScheduleConstant}
	started := time.Now()
	trainer, err := New(t.Context(), spec, options, func(name string) ([]float32, error) {
		tensor, ok := source.Tensors[name]
		if !ok {
			return nil, context.Canceled
		}
		return safetensors.ReadF32(tensor)
	}, InitialAdapter(spec, options, 42), config)
	testcheck.NoError(t, err)
	defer trainer.Close()
	t.Logf("placed in %.1fs", time.Since(started).Seconds())
	for _, extent := range []int{32, 64} {
		geometry := Geometry{Height: extent, Width: extent, Text: 64}
		if extent == 32 {
			testcheck.NoError(t, trainer.Reserve(t.Context(), Geometry{Height: 64, Width: 64, Text: 64}))
		}
		stream := rand.New(rand.NewPCG(1, 2))
		values := func(count int) []float32 {
			out := make([]float32, count)
			for index := range out {
				out[index] = float32(stream.NormFloat64())
			}
			return out
		}
		latent, noise := values(geometry.image()*spec.InChannels), values(geometry.image()*spec.InChannels)
		example := Example{Geometry: geometry, Latent: latent, Refined: values(geometry.Text * spec.Hidden)}
		noised, target := FlowPath(latent, noise, 0.5)
		testcheck.NoError(t, trainer.Clear())
		started = time.Now()
		loss, err := trainer.Accumulate(t.Context(), example, noised, target, 500)
		testcheck.NoError(t, err)
		forwardBackward := time.Since(started)
		norm, err := trainer.Gather(1)
		testcheck.NoError(t, err)
		started = time.Now()
		if _, err := trainer.Apply(loss, norm); err != nil {
			t.Fatal(err)
		}
		applied := time.Since(started)
		after, err := trainer.Loss(t.Context(), example, noised, target, 500)
		testcheck.NoError(t, err)
		var peak uint64
		if err := trainer.worker.Do(t.Context(), func(state *device.State) error {
			peak = state.Driver.MemoryStats().PeakBytes
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		t.Logf("picture %d square: forward and backward %.2fs, apply %.2fs, loss %.5f then %.5f, gradient %.4g, peak %.2f GiB", extent*16, forwardBackward.Seconds(), applied.Seconds(), loss, after, norm, float64(peak)/(1<<30))
	}
}
