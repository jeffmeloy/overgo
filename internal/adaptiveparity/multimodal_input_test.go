package adaptiveparity

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/projector"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

type e4bVisionGolden struct {
	PatchCount int                    `json:"n_patch"`
	Layers     map[string][][]float64 `json:"layers"`
	SoftTokens [][]float64            `json:"soft_tokens"`
	SoftCount  int                    `json:"n_soft_tokens"`
	SoftDim    int                    `json:"soft_dim"`
}

func TestMultimodalInputMatrix(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv("OVERGO_CUDA_TEST") != "1" {
		t.Skip("set OVERGO_CUDA_TEST=1 for real multimodal parity")
	}
	t.Run("gemma-e4b-image", testGemmaE4BImageParity)
}

func testGemmaE4BImageParity(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	imagePath := testutil.FixturePath(t, "e4b_vision", "gemma4_mm_image.png")
	goldenPath := testutil.FixturePath(t, "e4b_vision", "g_vision_f32.json")
	assertSHA256(t, projectorPath, "1e5580d6d8b0beeaf2aad9b3c29a61ce62cb57e47965b8a95f26378c216935db")
	assertSHA256(t, imagePath, "996fea5cea787f3abb6d1377fc88642ade6bdb8dc9a7b9e6ad909e58687b776e")
	assertSHA256(t, goldenPath, "f0b1c6f15a5432f5d4e3fcfd6c95c9292c58b0e29a3e8c4b63ff2a9f9ea3c35b")
	goldenRaw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: E4B vision golden absent; parity NOT verified: %v", err)
	}
	var golden e4bVisionGolden
	if err := json.Unmarshal(goldenRaw, &golden); err != nil {
		t.Fatal(err)
	}
	imageFile, err := os.Open(imagePath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: E4B image corpus absent; parity NOT verified: %v", err)
	}
	source, err := png.Decode(imageFile)
	closeErr := imageFile.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	runner, err := projector.OpenGemma4TowerWithOptions(projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatalf("UNAVAILABLE: E4B projector absent or CUDA unavailable; parity NOT verified: %v", err)
	}
	defer func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	}()
	input, err := projector.PreprocessGemma4VisionTowerImage(source, runner.Spec().Vision)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	output, trace, err := runner.EncodeVisionPatchesTrace(context.Background(), input.PixelValues, input.Positions)
	if err != nil {
		t.Fatal(err)
	}
	wall := time.Since(start)
	if wall > 2*time.Second {
		t.Fatalf("E4B resident vision wall = %s, limit 2s", wall)
	}
	if output.PatchCount != golden.PatchCount || output.SoftTokens != golden.SoftCount ||
		int(output.Embeddings.Shape.Dims[0]) != golden.SoftDim {
		t.Fatalf("E4B vision shape = patches %d, soft [%d,%d], want %d [%d,%d]",
			output.PatchCount, output.SoftTokens, output.Embeddings.Shape.Dims[0],
			golden.PatchCount, golden.SoftCount, golden.SoftDim)
	}
	worstName, worstRelative := "", 0.0
	stageNames := []string{"patch_embed"}
	for layer := 0; layer < 16; layer++ {
		stageNames = append(stageNames, fmt.Sprintf("enc%d", layer))
	}
	stageNames = append(stageNames, "pooler")
	for _, name := range stageNames {
		want := golden.Layers[name]
		got, ok := trace.Stages[name]
		if !ok {
			t.Fatalf("E4B vision trace lacks %s", name)
		}
		relative := sampledRelative(got.Data, want, int(got.Shape.Dims[0]))
		if relative > worstRelative {
			worstName, worstRelative = name, relative
		}
		limit := 5e-3
		if name == "patch_embed" {
			limit = 5e-2
		}
		if relative > limit {
			t.Errorf("E4B vision %s relative = %.6g, limit %.6g", name, relative, limit)
		}
	}
	soft := trace.Stages["soft_tokens"]
	softRelative := sampledRelative(soft.Data, golden.SoftTokens, int(soft.Shape.Dims[0]))
	if softRelative > 5e-3 {
		t.Fatalf("E4B vision soft-token relative = %.6g, limit 0.005", softRelative)
	}
	t.Logf("E4B image parity: %d patches -> %d x %d soft tokens in %s; worst stage %s %.6g; soft %.6g",
		output.PatchCount, output.SoftTokens, output.Embeddings.Shape.Dims[0], wall, worstName, worstRelative, softRelative)
}

func assertSHA256(t *testing.T, path, want string) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("UNAVAILABLE: fingerprint input %s absent; parity NOT verified: %v", path, err)
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
	if got := fmt.Sprintf("%x", hash.Sum(nil)); got != want {
		t.Fatalf("evidence %s SHA-256 = %s, want %s", path, got, want)
	}
}

func sampledRelative(got []float32, want [][]float64, width int) float64 {
	worst := 0.0
	for row := range want {
		for column, expected := range want[row] {
			relative := math.Abs(float64(got[row*width+column])-expected) / (1 + math.Abs(expected))
			worst = max(worst, relative)
		}
	}
	return worst
}
