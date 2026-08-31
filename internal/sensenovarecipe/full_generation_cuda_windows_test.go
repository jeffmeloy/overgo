//go:build windows

package sensenovarecipe

import (
	"bufio"
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	_ "image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
)

type fullGenerationOracle struct {
	Schema string `json:"schema"`
	Source struct {
		PromptPath string `json:"prompt_path"`
		PromptSHA  string `json:"prompt_sha256"`
		SourceSHA  string `json:"source_sha256"`
	} `json:"source"`
	Request  GenerationRequest `json:"request"`
	Adaptive struct {
		WallSeconds float64 `json:"wall_seconds"`
		PeakBytes   uint64  `json:"peak_bytes"`
		PNG         string  `json:"reference_png_sha256"`
	} `json:"adaptive_native"`
	Overgo struct {
		PNG        string           `json:"reviewed_png_sha256"`
		Review     string           `json:"review"`
		Assertions []string         `json:"assertions"`
		Signature  fullImageQuality `json:"signature"`
	} `json:"overgo_quality"`
}

type fullGenerationPromptRow struct {
	Prompt string `json:"prompt"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Seed   int64  `json:"seed"`
}

func TestSenseNovaFullGenerationLeadership(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_SENSENOVA_FULL") != "1" {
		t.Skip("set OVERGO_SENSENOVA_FULL=1 for the 50-step quality gate")
	}
	var oracle fullGenerationOracle
	if err := jsonfile.Decode(filepath.Join("..", "..", "fixtures", "sensenova", "full_generation_quality.json"), &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != "overgo.sensenova-full-generation/v1" || oracle.Adaptive.WallSeconds <= 0 ||
		oracle.Adaptive.PeakBytes == 0 || len(oracle.Adaptive.PNG) != sha256.Size*2 || len(oracle.Overgo.PNG) != sha256.Size*2 ||
		oracle.Overgo.Review == "" || len(oracle.Overgo.Assertions) != 4 || len(oracle.Overgo.Signature.Grid) != 64 {
		t.Fatal("SenseNova full-generation oracle is incomplete")
	}
	prompt := loadFullGenerationPrompt(t, oracle)
	request := oracle.Request
	request.Prompt = prompt
	result := runProductionImageGeneration(t, request)
	hash := fmt.Sprintf("%x", sha256.Sum256(result.image.Data))
	if result.image.Width != request.Width || result.image.Height != request.Height || result.image.Channels != 3 {
		t.Fatalf("SenseNova full image=%dx%dx%d sha256=%s, want %dx%dx3",
			result.image.Width, result.image.Height, result.image.Channels, hash, request.Width, request.Height)
	}
	quality := measureFullImageQuality(t, result.image.Data)
	requireFullImageQuality(t, quality, oracle.Overgo.Signature)
	if result.wall.Seconds() >= oracle.Adaptive.WallSeconds {
		t.Fatalf("SenseNova full wall=%.3fs does not lead adaptive %.3fs", result.wall.Seconds(), oracle.Adaptive.WallSeconds)
	}
	if result.peakBytes >= oracle.Adaptive.PeakBytes {
		t.Fatalf("SenseNova full peak=%d does not lead adaptive %d", result.peakBytes, oracle.Adaptive.PeakBytes)
	}
	t.Logf("SenseNova full generation leadership: wall=%.3fs adaptive=%.3fs peak=%.3fGiB adaptive=%.3fGiB png=%s reviewed_exact=%v",
		result.wall.Seconds(), oracle.Adaptive.WallSeconds, float64(result.peakBytes)/(1<<30),
		float64(oracle.Adaptive.PeakBytes)/(1<<30), hash, hash == oracle.Overgo.PNG)
}

type fullImageQuality struct {
	Mean              float64   `json:"mean"`
	StandardDeviation float64   `json:"standard_deviation"`
	EdgeMean          float64   `json:"edge_mean"`
	BlueFraction      float64   `json:"blue_fraction"`
	OrangeFraction    float64   `json:"orange_fraction"`
	WhiteFraction     float64   `json:"white_fraction"`
	DarkFraction      float64   `json:"dark_fraction"`
	Grid              []float64 `json:"grid"`
	ScalarTolerance   float64   `json:"scalar_tolerance"`
	GridMAEMax        float64   `json:"grid_mae_max"`
	GridWorstMax      float64   `json:"grid_worst_max"`
}

func measureFullImageQuality(t testing.TB, data []byte) fullImageQuality {
	t.Helper()
	decoded, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	bounds := decoded.Bounds()
	pixels := bounds.Dx() * bounds.Dy()
	values := make([]float64, pixels)
	quality := fullImageQuality{Grid: make([]float64, 64)}
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r16, g16, b16, _ := decoded.At(x, y).RGBA()
			r, g, blue := float64(r16>>8), float64(g16>>8), float64(b16>>8)
			luminance := (0.2126*r + 0.7152*g + 0.0722*blue) / 255
			index := (y-bounds.Min.Y)*bounds.Dx() + x - bounds.Min.X
			values[index] = luminance
			quality.Mean += luminance
			cell := min(7, (y-bounds.Min.Y)*8/bounds.Dy())*8 + min(7, (x-bounds.Min.X)*8/bounds.Dx())
			quality.Grid[cell] += luminance
			if blue > 100 && blue > r*1.2 && blue > g*1.05 {
				quality.BlueFraction++
			}
			if r > 180 && g > 50 && g < 190 && blue < 100 {
				quality.OrangeFraction++
			}
			if r > 230 && g > 230 && blue > 230 {
				quality.WhiteFraction++
			}
			if r < 70 && g < 70 && blue < 70 {
				quality.DarkFraction++
			}
		}
	}
	count := float64(pixels)
	quality.Mean /= count
	for _, value := range values {
		delta := value - quality.Mean
		quality.StandardDeviation += delta * delta
	}
	quality.StandardDeviation = math.Sqrt(quality.StandardDeviation / count)
	var edges float64
	for y := range bounds.Dy() {
		for x := range bounds.Dx() {
			index := y*bounds.Dx() + x
			if x > 0 {
				quality.EdgeMean += math.Abs(values[index] - values[index-1])
				edges++
			}
			if y > 0 {
				quality.EdgeMean += math.Abs(values[index] - values[index-bounds.Dx()])
				edges++
			}
		}
	}
	quality.EdgeMean /= edges
	cellPixels := count / 64
	for index := range quality.Grid {
		quality.Grid[index] /= cellPixels
	}
	quality.BlueFraction /= count
	quality.OrangeFraction /= count
	quality.WhiteFraction /= count
	quality.DarkFraction /= count
	return quality
}

func requireFullImageQuality(t testing.TB, got, want fullImageQuality) {
	t.Helper()
	scalars := []struct {
		name      string
		got, want float64
	}{
		{"mean", got.Mean, want.Mean}, {"standard deviation", got.StandardDeviation, want.StandardDeviation},
		{"edge mean", got.EdgeMean, want.EdgeMean}, {"blue fraction", got.BlueFraction, want.BlueFraction},
		{"orange fraction", got.OrangeFraction, want.OrangeFraction}, {"white fraction", got.WhiteFraction, want.WhiteFraction},
		{"dark fraction", got.DarkFraction, want.DarkFraction},
	}
	if want.ScalarTolerance <= 0 || want.GridMAEMax <= 0 || want.GridWorstMax <= 0 {
		t.Fatal("SenseNova quality tolerances are incomplete")
	}
	for _, scalar := range scalars {
		if math.Abs(scalar.got-scalar.want) > want.ScalarTolerance {
			t.Fatalf("SenseNova quality %s=%g want %g +/- %g", scalar.name, scalar.got, scalar.want, want.ScalarTolerance)
		}
	}
	mae, worst := 0.0, 0.0
	for index := range want.Grid {
		delta := math.Abs(got.Grid[index] - want.Grid[index])
		mae += delta
		worst = max(worst, delta)
	}
	mae /= float64(len(want.Grid))
	if mae > want.GridMAEMax || worst > want.GridWorstMax {
		t.Fatalf("SenseNova spatial quality MAE=%g worst=%g, limits=%g/%g", mae, worst, want.GridMAEMax, want.GridWorstMax)
	}
	t.Logf("SenseNova reviewed quality: mean=%.4f std=%.4f edge=%.4f blue=%.4f orange=%.4f white=%.4f dark=%.4f grid_mae=%.4f grid_worst=%.4f",
		got.Mean, got.StandardDeviation, got.EdgeMean, got.BlueFraction, got.OrangeFraction, got.WhiteFraction, got.DarkFraction, mae, worst)
}

func loadFullGenerationPrompt(t testing.TB, oracle fullGenerationOracle) string {
	t.Helper()
	path := filepath.Join(senseNovaModelDir, filepath.FromSlash(oracle.Source.PromptPath))
	if got := fullGenerationFileSHA(t, path); got != oracle.Source.SourceSHA {
		t.Fatalf("SenseNova prompt source sha256=%s, want %s", got, oracle.Source.SourceSHA)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var row fullGenerationPromptRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		hash := fmt.Sprintf("%x", sha256.Sum256([]byte(row.Prompt)))
		if hash != oracle.Source.PromptSHA {
			continue
		}
		seed := row.Seed
		seed = cmp.Or(seed, 42)
		if row.Width != oracle.Request.Width || row.Height != oracle.Request.Height || seed != oracle.Request.Seed {
			t.Fatalf("SenseNova prompt request=%dx%d seed=%d, want %dx%d seed=%d",
				row.Width, row.Height, seed, oracle.Request.Width, oracle.Request.Height, oracle.Request.Seed)
		}
		return row.Prompt
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	t.Fatalf("SenseNova prompt %s absent from %s", oracle.Source.PromptSHA, path)
	return ""
}

func fullGenerationFileSHA(t testing.TB, path string) string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	_, copyErr := io.Copy(hash, file)
	closeErr := file.Close()
	if copyErr != nil {
		t.Fatal(copyErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	return hex.EncodeToString(hash.Sum(nil))
}
