//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

func TestGeneratorRealCheckpointBaseline(t *testing.T) {
	requireLongTest(t)
	if os.Getenv("OVERGO_KREA_BASELINE") != "1" {
		t.Skip("set OVERGO_KREA_BASELINE=1 to measure the real Krea pipeline")
	}
	request := Request{
		Prompt: "a red fox licking a vanilla ice cream cone in snow",
		Width:  256, Height: 256, Steps: 8, Seed: 42, DynamicShiftMu: 1.15,
	}
	ctx := context.Background()
	totalStart := time.Now()
	generator, err := LoadGenerator(ctx, kreaModelDir, request)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := generator.Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
	}()
	loadWall := time.Since(totalStart)

	prepareStart := time.Now()
	session, err := generator.prepare(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	prepareWall := time.Since(prepareStart)
	wantMedian := [...]float64{-0.003021240234375, -0.00518798828125, -0.008575439453125, -0.009796142578125, -0.011077880859375, -0.008880615234375, -0.0128173828125, -0.04833984375}
	wantMAD := [...]float64{0.641510009765625, 0.607421875, 0.569549560546875, 0.517608642578125, 0.455718994140625, 0.389556884765625, 0.3172607421875, 0.29736328125}
	wantVelocityMedian := [...]float64{0.035400390625, 0.069580078125, 0.048828125, 0.02197265625, 0.010162353515625, 0.0062713623046875, 0.0068359375, 0.01806640625}
	wantVelocityMAD := [...]float64{0.777099609375, 0.828857421875, 0.787109375, 0.74853515625, 0.728515625, 0.7093963623046875, 0.7001953125, 0.66162109375}
	prior, err := generator.pipeline.Latent(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var integrateWall time.Duration
	for step := range generator.schedule.Steps {
		started := time.Now()
		if err := generator.pipeline.Advance(ctx, generator.schedule.Sigmas[step], generator.schedule.Deltas[step]); err != nil {
			t.Fatal(err)
		}
		integrateWall += time.Since(started)
		latent, err := generator.pipeline.Latent(ctx)
		if err != nil {
			t.Fatal(err)
		}
		median, mad := medianMAD(latent)
		t.Logf("step %d latent median/MAD: got %.6f/%.6f adaptive %.6f/%.6f", step, median, mad, wantMedian[step], wantMAD[step])
		if math.Abs(median-wantMedian[step]) > 0.005 || math.Abs(mad-wantMAD[step]) > 0.01 {
			t.Fatalf("step %d latent distribution exceeds adaptive bound", step)
		}
		velocity := make([]float32, len(latent))
		for index := range velocity {
			velocity[index] = (latent[index] - prior[index]) / float32(generator.schedule.Deltas[step])
		}
		velocityMedian, velocityMAD := medianMAD(velocity)
		t.Logf("step %d velocity median/MAD: got %.6f/%.6f adaptive %.6f/%.6f", step, velocityMedian, velocityMAD, wantVelocityMedian[step], wantVelocityMAD[step])
		if math.Abs(velocityMedian-wantVelocityMedian[step]) > 0.012 || math.Abs(velocityMAD-wantVelocityMAD[step]) > 0.015 {
			t.Fatalf("step %d velocity distribution exceeds adaptive bound", step)
		}
		prior = latent
	}
	generator.integrated = true
	decodeStart := time.Now()
	image, err := generator.decode(ctx, session)
	if err != nil {
		t.Fatal(err)
	}
	decodeWall := time.Since(decodeStart)
	if image.Channels != 3 || image.Height != 256 || image.Width != 256 || len(image.Pixels) != 3*256*256 {
		t.Fatalf("image geometry = %d/%dx%d/%d", image.Channels, image.Width, image.Height, len(image.Pixels))
	}
	encoded := make([]byte, len(image.Pixels))
	minimum, maximum := float32(math.Inf(1)), float32(math.Inf(-1))
	golden, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "krea", "raw", "g3_decoded_rgb_f32le.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(golden) != 4*len(image.Pixels) {
		t.Fatalf("golden bytes = %d, want %d", len(golden), 4*len(image.Pixels))
	}
	var absoluteError, squaredError float64
	for index, value := range image.Pixels {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("pixel %d is not finite", index)
		}
		minimum = min(minimum, value)
		maximum = max(maximum, value)
		normalized := min(max((value+1)*0.5, 0), 1)
		encoded[index] = byte(math.Round(float64(normalized * 255)))
		want := math.Float32frombits(binary.LittleEndian.Uint32(golden[4*index:]))
		delta := float64(normalized - want)
		absoluteError += math.Abs(delta)
		squaredError += delta * delta
	}
	if maximum-minimum < 0.1 {
		t.Fatalf("image is degenerate: range [%g,%g]", minimum, maximum)
	}
	var memory driver.MemoryStats
	if err := generator.pipeline.runtime.worker.Do(ctx, func(state *device.State) error {
		memory = state.Driver.MemoryStats()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(encoded)
	totalWall := time.Since(totalStart)
	mae := absoluteError / float64(len(image.Pixels))
	rmse := math.Sqrt(squaredError / float64(len(image.Pixels)))
	t.Logf(
		"Krea 256 seed42 load=%.3fs prepare=%.3fs integrate=%.3fs decode=%.3fs total=%.3fs peak_device=%.3fGiB resident=%.3fGiB u8_sha256=%s range=[%g,%g]",
		loadWall.Seconds(), prepareWall.Seconds(), integrateWall.Seconds(), decodeWall.Seconds(), totalWall.Seconds(),
		float64(memory.PeakBytes)/(1<<30), float64(generator.pipeline.ResidentBytes())/(1<<30),
		hex.EncodeToString(hash[:]), minimum, maximum,
	)
	t.Logf("adaptive golden delta: MAE=%.6f RMSE=%.6f", mae, rmse)
	if mae > 0.04 || rmse > 0.07 {
		t.Fatalf("adaptive golden delta exceeds bound: MAE=%.6f RMSE=%.6f", mae, rmse)
	}
	const adaptiveWall = 25_430_488_300 * time.Nanosecond
	if totalWall >= adaptiveWall {
		t.Fatalf("Krea wall %s does not beat adaptive %s", totalWall, adaptiveWall)
	}
}

func medianMAD(values []float32) (float64, float64) {
	ordered := append([]float32(nil), values...)
	slices.Sort(ordered)
	median := ordered[len(ordered)/2]
	deviations := make([]float32, len(ordered))
	for index, value := range ordered {
		deviations[index] = float32(math.Abs(float64(value - median)))
	}
	slices.Sort(deviations)
	return float64(median), float64(deviations[len(deviations)/2])
}
