//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

const krea2048AdaptivePNG = "b25fa024af3b623ee443c74207b6a6ee32abeb6d0a5df7ba61cc6147c8522a5c"

const krea2048AdaptivePNGPath = `C:\Users\jeffm\adaptive_new\.claude\worktrees\image_gen\.media_artifacts\image_artifacts\krea2_native_go_fox_seed42.png`

func BenchmarkKrea2048Leadership(b *testing.B) {
	if os.Getenv("OVERGO_CUDA_TEST") != "1" {
		b.Skip("set OVERGO_CUDA_TEST=1")
	}
	if _, err := os.Stat(kreaModelDir + `\model_index.json`); err != nil {
		b.Fatalf("Krea artifact unavailable: %v", err)
	}
	request := Request{
		Prompt: kreaGoldenPrompt,
		Width:  2048, Height: 2048, Steps: 8, Seed: 42, DynamicShiftMu: 1.15,
	}
	for b.Loop() {
		benchmarkKrea2048(b, request)
	}
}

func benchmarkKrea2048(b *testing.B, request Request) {
	b.Helper()
	ctx := context.Background()
	totalStart := time.Now()
	profile, err := ResolveProfile(kreaModelDir)
	if err != nil {
		b.Fatal(err)
	}
	generator, err := LoadGenerator(ctx, kreaModelDir, profile, request)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := generator.Close(ctx); err != nil {
			b.Errorf("close: %v", err)
		}
	}()
	loadWall := time.Since(totalStart)

	prepareStart := time.Now()
	session, err := generator.prepare(ctx, request)
	if err != nil {
		b.Fatal(err)
	}
	prepareWall := time.Since(prepareStart)
	integrateStart := time.Now()
	if _, err := generator.integrate(ctx, session); err != nil {
		b.Fatal(err)
	}
	integrateWall := time.Since(integrateStart)
	decodeStart := time.Now()
	generated, err := generator.decode(ctx, session)
	if err != nil {
		b.Fatal(err)
	}
	decodeWall := time.Since(decodeStart)
	encodeStart := time.Now()
	encoded, minimum, maximum, err := encodeAdaptivePNG(generated)
	if err != nil {
		b.Fatal(err)
	}
	encodeWall := time.Since(encodeStart)
	var memory driver.MemoryStats
	if err := generator.pipeline.runtime.worker.Do(ctx, func(state *device.State) error {
		memory = state.Driver.MemoryStats()
		return nil
	}); err != nil {
		b.Fatal(err)
	}
	totalWall := time.Since(totalStart)
	hash := fmt.Sprintf("%x", sha256.Sum256(encoded))
	mae, rmse, err := compareAdaptivePNG(encoded, krea2048AdaptivePNGPath)
	if err != nil {
		b.Fatal(err)
	}
	b.Logf(
		"Krea 2048 seed42 load=%.3fs prepare=%.3fs integrate=%.3fs decode=%.3fs png=%.3fs total=%.3fs peak_device=%.3fGiB resident=%.3fGiB png_sha256=%s range=[%g,%g] MAE=%.6f RMSE=%.6f",
		loadWall.Seconds(), prepareWall.Seconds(), integrateWall.Seconds(), decodeWall.Seconds(), encodeWall.Seconds(), totalWall.Seconds(),
		float64(memory.PeakBytes)/(1<<30), float64(generator.pipeline.ResidentBytes())/(1<<30), hash, minimum, maximum, mae, rmse,
	)
	if mae > 0.04 {
		b.Fatalf("adaptive image MAE exceeds bound: %.6f", mae)
	}
	const adaptiveWall = 158_144 * time.Millisecond
	if totalWall >= adaptiveWall {
		b.Fatalf("Krea wall %s does not beat adaptive %s", totalWall, adaptiveWall)
	}
	const adaptivePeak = uint64(33_468_413_444)
	if memory.PeakBytes >= adaptivePeak {
		b.Fatalf("Krea peak = %d, want below adaptive %d", memory.PeakBytes, adaptivePeak)
	}
}

func compareAdaptivePNG(generated []byte, goldenPath string) (float64, float64, error) {
	wantBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		return 0, 0, fmt.Errorf("read adaptive PNG: %w", err)
	}
	if hash := fmt.Sprintf("%x", sha256.Sum256(wantBytes)); hash != krea2048AdaptivePNG {
		return 0, 0, fmt.Errorf("adaptive PNG sha256 = %s, want %s", hash, krea2048AdaptivePNG)
	}
	got, err := png.Decode(bytes.NewReader(generated))
	if err != nil {
		return 0, 0, fmt.Errorf("decode generated PNG: %w", err)
	}
	want, err := png.Decode(bytes.NewReader(wantBytes))
	if err != nil {
		return 0, 0, fmt.Errorf("decode adaptive PNG: %w", err)
	}
	if got.Bounds() != want.Bounds() {
		return 0, 0, fmt.Errorf("PNG bounds = %v, want %v", got.Bounds(), want.Bounds())
	}
	var absoluteError, squaredError float64
	for y := got.Bounds().Min.Y; y < got.Bounds().Max.Y; y++ {
		for x := got.Bounds().Min.X; x < got.Bounds().Max.X; x++ {
			gotRGBA := [...]uint32{0, 0, 0}
			wantRGBA := [...]uint32{0, 0, 0}
			gotRGBA[0], gotRGBA[1], gotRGBA[2], _ = got.At(x, y).RGBA()
			wantRGBA[0], wantRGBA[1], wantRGBA[2], _ = want.At(x, y).RGBA()
			for channel := range gotRGBA {
				delta := float64(int64(gotRGBA[channel])-int64(wantRGBA[channel])) / 65535
				absoluteError += math.Abs(delta)
				squaredError += delta * delta
			}
		}
	}
	count := float64(got.Bounds().Dx() * got.Bounds().Dy() * 3)
	return absoluteError / count, math.Sqrt(squaredError / count), nil
}

func encodeAdaptivePNG(generated Image) ([]byte, float32, float32, error) {
	if generated.Channels != 3 || generated.Width <= 0 || generated.Height <= 0 {
		return nil, 0, 0, fmt.Errorf("image geometry = %d/%dx%d", generated.Channels, generated.Width, generated.Height)
	}
	plane := generated.Width * generated.Height
	if len(generated.Pixels) != generated.Channels*plane {
		return nil, 0, 0, fmt.Errorf("image pixels = %d, want %d", len(generated.Pixels), generated.Channels*plane)
	}
	output := image.NewRGBA(image.Rect(0, 0, generated.Width, generated.Height))
	minimum, maximum := float32(math.Inf(1)), float32(math.Inf(-1))
	pixelByte := func(value float32) (uint8, error) {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, errors.New("image contains non-finite pixels")
		}
		minimum = min(minimum, value)
		maximum = max(maximum, value)
		scaled := (float64(value) + 1) * 127.5
		return uint8(min(max(scaled, 0), 255)), nil
	}
	for position := range plane {
		offset := position * 4
		for channel := range 3 {
			value, err := pixelByte(generated.Pixels[channel*plane+position])
			if err != nil {
				return nil, 0, 0, err
			}
			output.Pix[offset+channel] = value
		}
		output.Pix[offset+3] = 255
	}
	if maximum-minimum < 0.1 {
		return nil, 0, 0, fmt.Errorf("image is degenerate: range [%g,%g]", minimum, maximum)
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, output); err != nil {
		return nil, 0, 0, err
	}
	return encoded.Bytes(), minimum, maximum, nil
}
