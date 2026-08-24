package adaptiveparity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

type unlimitedOCRRow struct {
	Kind        string
	Coordinates [4]int
	Text        string
}

func TestUnlimitedOCRLayoutMetrics(t *testing.T) {
	rows, err := parseUnlimitedOCRRows("title [1, 2, 3, 4]alpha\ntext [5, 6, 7, 8]beta")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[1].Kind != "text" || rows[1].Coordinates != [4]int{5, 6, 7, 8} || rows[1].Text != "beta" {
		t.Fatalf("rows = %+v", rows)
	}
	if distance := runeEditDistance("Line 17.", "Line 17"); distance != 1 {
		t.Fatalf("edit distance = %d", distance)
	}
}

func TestUnlimitedOCRProductionParity(t *testing.T) {
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
	assertSHA256(t, projectorPath, "894b6ce68b553a2f176b1d69099b42e369a63402274ccd8c19614466ea85d2ea")
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
	vision, err := projector.OpenAs[*projector.DeepSeekOCRRunner](context.Background(), projectorPath, projector.OpenOptions{
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
	language, err := inference.OpenWithProgram(context.Background(), &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer language.Close()
	if golden.Decode.Prompt != projector.DeepSeekOCRImagePad+"document parsing." {
		t.Fatalf("Unlimited OCR oracle prompt = %q", golden.Decode.Prompt)
	}
	projectStarted := time.Now()
	prompt, err := mustProjectorSession(t, vision).BuildImagePrompt(
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
	ids, projected, err := inference.CompileProjectedInputs(prompt, language.Spec().EmbeddingLength)
	if err != nil {
		t.Fatal(err)
	}
	greedy, err := sampling.New(sampling.Config{
		Temperature: 0, TopK: 1, TopP: 1, NoRepeatNgramSize: 35, NgramWindow: 128,
	})
	if err != nil {
		t.Fatal(err)
	}
	generateStarted := time.Now()
	generated, _, err := language.Generate(context.Background(), "", inference.GenerateOptions{
		MaxNewTokens: len(golden.GeneratedTokenIDs), Sampler: greedy, DeviceGreedy: true,
		PromptTokenIDs: ids, ProjectedInputs: &projected,
	})
	if err != nil {
		t.Fatal(err)
	}
	generated = generated[len(ids):]
	wantGenerated := golden.GeneratedTokenIDs
	const exactPrefixTokens = 200
	if len(generated) < exactPrefixTokens || len(wantGenerated) < exactPrefixTokens ||
		!slices.Equal(generated[:exactPrefixTokens], wantGenerated[:exactPrefixTokens]) {
		t.Fatal("Unlimited OCR exact token prefix differs")
	}
	if len(generated) == 0 || len(wantGenerated) == 0 || generated[len(generated)-1] != wantGenerated[len(wantGenerated)-1] {
		t.Fatal("Unlimited OCR terminal token differs")
	}
	if delta := len(generated) - len(wantGenerated); delta < -1 || delta > 1 {
		t.Fatalf("Unlimited OCR output tokens = %d, want %d±1", len(generated), len(wantGenerated))
	}
	text, err := language.Detokenize(generated, inference.RenderText)
	if err != nil {
		t.Fatal(err)
	}
	wantText, err := language.Detokenize(wantGenerated, inference.RenderText)
	if err != nil {
		t.Fatal(err)
	}
	gotRows, err := parseUnlimitedOCRRows(text)
	if err != nil {
		t.Fatal(err)
	}
	wantRows, err := parseUnlimitedOCRRows(wantText)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotRows) != 29 || len(wantRows) != 29 {
		t.Fatalf("Unlimited OCR rows = %d, want %d", len(gotRows), len(wantRows))
	}
	for index := range wantRows {
		if gotRows[index].Kind != wantRows[index].Kind {
			t.Fatalf("Unlimited OCR row %d kind = %q, want %q", index, gotRows[index].Kind, wantRows[index].Kind)
		}
		for axis := range gotRows[index].Coordinates {
			delta := gotRows[index].Coordinates[axis] - wantRows[index].Coordinates[axis]
			if delta < -1 || delta > 1 {
				t.Fatalf("Unlimited OCR row %d coordinate %d delta = %d", index, axis, delta)
			}
		}
		if distance := runeEditDistance(gotRows[index].Text, wantRows[index].Text); distance > 1 {
			t.Fatalf("Unlimited OCR row %d text edit distance = %d", index, distance)
		}
	}
	t.Logf("Unlimited OCR production parity: 273 image tokens; exact %d-token prefix; %d bounded rows; project %s; generate %s",
		exactPrefixTokens, len(gotRows),
		projectDuration.Round(time.Millisecond), time.Since(generateStarted).Round(time.Millisecond))
}

func parseUnlimitedOCRRows(text string) ([]unlimitedOCRRow, error) {
	lines := strings.Split(strings.TrimSpace(text), "\n")
	rows := make([]unlimitedOCRRow, len(lines))
	for index, line := range lines {
		head, body, ok := strings.Cut(line, "]")
		if !ok {
			return nil, fmt.Errorf("Unlimited OCR row %d lacks coordinates", index)
		}
		row := unlimitedOCRRow{Text: body}
		matched, err := fmt.Sscanf(head, "%s [%d, %d, %d, %d",
			&row.Kind, &row.Coordinates[0], &row.Coordinates[1], &row.Coordinates[2], &row.Coordinates[3])
		if err != nil || matched != 5 {
			return nil, fmt.Errorf("Unlimited OCR row %d header is invalid", index)
		}
		rows[index] = row
	}
	return rows, nil
}

func runeEditDistance(leftText, rightText string) int {
	left, right := []rune(leftText), []rune(rightText)
	row := make([]int, len(right)+1)
	for index := range row {
		row[index] = index
	}
	for leftIndex, leftRune := range left {
		previous := row[0]
		row[0] = leftIndex + 1
		for rightIndex, rightRune := range right {
			above := row[rightIndex+1]
			cost := 0
			if leftRune != rightRune {
				cost = 1
			}
			row[rightIndex+1] = min(row[rightIndex+1]+1, row[rightIndex]+1, previous+cost)
			previous = above
		}
	}
	return row[len(right)]
}
