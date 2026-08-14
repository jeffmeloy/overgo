package adaptiveparity

import (
	"context"
	"encoding/json"
	"errors"
	"image"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type unlimitedOCRGolden struct {
	InputIDs          []tokenizer.TokenID `json:"input_ids"`
	GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
	OutputText        string              `json:"output_text"`
	Decode            struct {
		Prompt string `json:"prompt"`
	} `json:"decode"`
}

func TestUnlimitedOCRProductionPrefixParity(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv("OVERGO_CUDA_TEST") != "1" {
		t.Skip("set OVERGO_CUDA_TEST=1 for real Unlimited OCR parity")
	}
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	fixtureRoot := filepath.Join(filepath.Dir(roots.Models), "fixtures")
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Unlimited-OCR-bf16.gguf")
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Unlimited-OCR-mmproj-bf16.gguf")
	imagePath := filepath.Join(fixtureRoot, "unlimitedocr_image.png")
	goldenPath := filepath.Join(fixtureRoot, "unlimitedocr_golden.json")
	assertSHA256(t, modelPath, "d4d2dff7c1988e5eb6900599158f78cf04e836fbf957176d186d2c97edf5c8b9")
	assertSHA256(t, projectorPath, "c6251635fe1b6d52195dfd886738f2f79ea6482cf8e2ad91a8b55b070e783ca7")
	assertSHA256(t, imagePath, "98db361580c88252e3942dd3ab197e332348ee0e295e5b22ec734de58c6f46e1")
	assertSHA256(t, goldenPath, "72339f0f4aece7c793d8c77a0c804ba2b121af3caf96340bf850d89bb744acd5")

	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	var golden unlimitedOCRGolden
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.InputIDs) == 0 || len(golden.GeneratedTokenIDs) == 0 || golden.OutputText == "" {
		t.Fatal("Unlimited OCR oracle is incomplete")
	}
	const generatedPrefixTokens = 128
	if len(golden.GeneratedTokenIDs) < generatedPrefixTokens {
		t.Fatal("Unlimited OCR oracle token prefix is incomplete")
	}
	lines := strings.Split(strings.TrimSpace(golden.OutputText), "\n")
	if len(lines) != 29 || !strings.Contains(lines[0], "ADAPTIVE GPT OCR LONG FIXTURE") ||
		!strings.HasSuffix(lines[28], "Line 28 adaptive memory scalable OCR validation row 0476") {
		t.Fatal("Unlimited OCR oracle layout is invalid")
	}
	imageFile, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	source, _, decodeErr := image.Decode(imageFile)
	closeErr := imageFile.Close()
	if decodeErr != nil || closeErr != nil {
		t.Fatal(errors.Join(decodeErr, closeErr))
	}
	vision, err := projector.OpenDeepSeekOCRWithOptions(projectorPath, projector.OpenOptions{
		CUDA: true, DisableDynamicTiles: true,
	})
	if err != nil {
		t.Fatalf("UNAVAILABLE: Unlimited OCR projector or CUDA absent; parity NOT verified: %v", err)
	}
	defer vision.Close()
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionRequest, recipe.ResidencyHybridNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	language, err := inference.OpenWithProgram(&loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer language.Close()
	if golden.Decode.Prompt != projector.DeepSeekOCRImagePad+"document parsing." {
		t.Fatalf("Unlimited OCR oracle prompt = %q", golden.Decode.Prompt)
	}
	projectStarted := time.Now()
	prompt, err := vision.BuildImagePrompt(
		context.Background(), language, source, "", "document parsing.", false,
	)
	if err != nil {
		t.Fatal(err)
	}
	projectDuration := time.Since(projectStarted)
	if !slices.Equal(prompt.TokenIDs, golden.InputIDs) {
		t.Fatalf("Unlimited OCR prompt IDs = %d, want %d", len(prompt.TokenIDs), len(golden.InputIDs))
	}
	if prompt.EmbeddingWidth != 1280 || len(prompt.EmbeddingTokenIndices) != 273 ||
		len(prompt.Embeddings) != 273*1280 {
		t.Fatalf("Unlimited OCR projection = width %d, tokens %d, values %d",
			prompt.EmbeddingWidth, len(prompt.EmbeddingTokenIndices), len(prompt.Embeddings))
	}
	for index, value := range prompt.Embeddings {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("Unlimited OCR projection value %d is non-finite", index)
		}
	}
	ids, projected, err := inference.ProjectedInputsForPrompt(language, prompt)
	if err != nil {
		t.Fatal(err)
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0, TopK: 1, TopP: 1})
	if err != nil {
		t.Fatal(err)
	}
	generateStarted := time.Now()
	generated, _, err := language.Generate(context.Background(), "", inference.GenerateOptions{
		MaxNewTokens: generatedPrefixTokens, Sampler: greedy, DeviceGreedy: true,
		PromptTokenIDs: ids, ProjectedInputs: &projected,
	})
	if err != nil {
		t.Fatal(err)
	}
	generated = generated[len(ids):]
	wantGenerated := golden.GeneratedTokenIDs[:generatedPrefixTokens]
	if !slices.Equal(generated, wantGenerated) {
		limit := min(len(generated), len(wantGenerated))
		mismatch := limit
		for index := range limit {
			if generated[index] != wantGenerated[index] {
				mismatch = index
				break
			}
		}
		t.Fatalf("Unlimited OCR prefix differs at %d: got %d tokens, want %d", mismatch, len(generated), len(wantGenerated))
	}
	text, err := language.DetokenizeTokens(generated)
	if err != nil {
		t.Fatal(err)
	}
	wantText, err := language.DetokenizeTokens(wantGenerated)
	if err != nil {
		t.Fatal(err)
	}
	if text != wantText {
		t.Fatalf("Unlimited OCR prefix text differs: got %d bytes, want %d", len(text), len(wantText))
	}
	t.Logf("Unlimited OCR production parity: 273 image tokens; exact %d-token prefix; 29-row oracle; project %s; generate %s",
		generatedPrefixTokens,
		projectDuration.Round(time.Millisecond), time.Since(generateStarted).Round(time.Millisecond))
}
