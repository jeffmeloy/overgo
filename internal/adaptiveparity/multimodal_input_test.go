package adaptiveparity

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/tensor"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type e4bVisionGolden struct {
	PatchCount int                    `json:"n_patch"`
	Layers     map[string][][]float64 `json:"layers"`
	SoftTokens [][]float64            `json:"soft_tokens"`
	SoftCount  int                    `json:"n_soft_tokens"`
	SoftDim    int                    `json:"soft_dim"`
}

type e4bAudioGolden struct {
	InputShape []int                  `json:"input_features_shape"`
	Layers     map[string][][]float64 `json:"layers"`
	SoftTokens [][]float64            `json:"soft_tokens"`
	SoftCount  int                    `json:"n_soft_tokens"`
	SoftDim    int                    `json:"soft_dim"`
}

type e4bImageLanguageGolden struct {
	Prefix     []tokenizer.TokenID `json:"prefix"`
	ImageToken tokenizer.TokenID   `json:"image_token"`
	ImageCount int                 `json:"image_count"`
	Suffix     []tokenizer.TokenID `json:"suffix"`
	NextToken  tokenizer.TokenID   `json:"next_token"`
	NextPiece  string              `json:"next_piece"`
}

type qwen35Probe struct {
	Shape      []int     `json:"shape"`
	ProbeIndex []int     `json:"probe_index"`
	ProbeValue []float32 `json:"probe_value"`
	L2         float64   `json:"l2"`
}

type qwen35VisionGolden struct {
	Grid             []int               `json:"image_grid_thw"`
	Tokens           int                 `json:"num_image_tokens"`
	InputIDs         []tokenizer.TokenID `json:"input_ids"`
	Pixels           qwen35Probe         `json:"pixel_values"`
	GeneratedTokenID []tokenizer.TokenID `json:"generated_token_ids"`
	Intermediates    struct {
		Merger qwen35Probe `json:"merger"`
	} `json:"vit_intermediates"`
}

type qwen35VideoGolden struct {
	Grid             []int               `json:"video_grid_thw"`
	Tokens           int                 `json:"num_video_tokens"`
	InputIDs         []tokenizer.TokenID `json:"input_ids"`
	PromptText       string              `json:"prompt_text"`
	Question         string              `json:"question"`
	Pixels           qwen35Probe         `json:"pixel_values_videos"`
	GeneratedTokenID []tokenizer.TokenID `json:"generated_token_ids"`
	Intermediates    struct {
		Merger qwen35Probe `json:"merger"`
	} `json:"vit_intermediates"`
}

func TestMultimodalInputMatrix(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv("OVERGO_CUDA_TEST") != "1" {
		t.Skip("set OVERGO_CUDA_TEST=1 for real multimodal parity")
	}
	t.Run("qwen35-image-video-language", testQwen35ImageVideoParity)
	t.Run("gemma4-12b-image-audio-video", testGemma4InputParity)
	t.Run("gemma-e4b-image", testGemmaE4BImageParity)
	t.Run("gemma-e4b-image-language", testGemmaE4BImageLanguageParity)
	t.Run("gemma-e4b-dynamic-resize", testGemmaE4BResizeParity)
	t.Run("gemma-e4b-video-order", testGemmaE4BVideoOrder)
	t.Run("gemma-e4b-audio", testGemmaE4BAudioParity)
}

func testGemma4InputParity(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	fixtureRoot := filepath.Join(filepath.Dir(roots.Models), "fixtures")
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-12B-it-fp8-mmproj-bf16.gguf")
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-12B-it-fp8-native.gguf")
	imagePath := filepath.Join(fixtureRoot, "gemma4_mm_image.png")
	imageGoldenPath := filepath.Join(fixtureRoot, "gemma4_mm_golden.json")
	wavePath := filepath.Join(fixtureRoot, "gemma4_audio_wave.f32")
	audioGoldenPath := filepath.Join(fixtureRoot, "gemma4_audio_golden.json")
	assertSHA256(t, projectorPath, "24489a6d072466fb288c43474ab9890e77eebe2503b0d7c514e626ee3a08e930")
	assertSHA256(t, modelPath, "64fb17f3c324dcbf9f83861e5d2ea1534a66ea0970223047e788a482b080853c")
	assertSHA256(t, imagePath, "996fea5cea787f3abb6d1377fc88642ade6bdb8dc9a7b9e6ad909e58687b776e")
	assertSHA256(t, imageGoldenPath, "f2e50cd5619ce5dfb92b91d8547daa930a5a9d974e01852e6fde81a0b2435d54")
	assertSHA256(t, wavePath, "98c8d1a25f96bbdfff5ca6153c9b18fad9740e300008618a970c18384b35d404")
	assertSHA256(t, audioGoldenPath, "273b9b00522ab9d8c8c74e5f333475f9c1bd70c50bf4f52d7c570b6a0fd2da97")
	var imageGolden struct {
		InputIDs          []tokenizer.TokenID `json:"input_ids"`
		GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
	}
	var audioGolden struct {
		InputIDs          []tokenizer.TokenID `json:"input_ids"`
		GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
		Steps             []struct {
			TopIDs []tokenizer.TokenID `json:"top_ids"`
		} `json:"steps"`
	}
	decodeJSONEvidence(t, imageGoldenPath, &imageGolden)
	decodeJSONEvidence(t, audioGoldenPath, &audioGolden)
	file, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	imageSource, decodeErr := png.Decode(file)
	closeErr := file.Close()
	if decodeErr != nil || closeErr != nil {
		t.Fatal(errors.Join(decodeErr, closeErr))
	}
	runner, err := projector.OpenAs[*projector.Gemma4Runner](context.Background(), projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatalf("UNAVAILABLE: Gemma4 projector or CUDA absent; parity NOT verified: %v", err)
	}
	defer runner.Close()
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionRequest, recipe.ResidencyDeviceNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	language, err := inference.OpenWithProgram(context.Background(), &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer language.Close()
	imagePrompt, err := mustProjectorSession(t, runner).BuildImagePrompt(
		context.Background(), language, imageSource, "", "What color dominates this image? One word.", false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(imagePrompt.TokenIDs, imageGolden.InputIDs) {
		t.Fatalf("Gemma4 image prompt IDs = %d, want %d", len(imagePrompt.TokenIDs), len(imageGolden.InputIDs))
	}
	imageToken := generatePromptFirstToken(t, language, imagePrompt)
	if len(imageGolden.GeneratedTokenIDs) == 0 || imageToken != imageGolden.GeneratedTokenIDs[0] {
		t.Fatalf("Gemma4 image first token = %d, want %v", imageToken, imageGolden.GeneratedTokenIDs)
	}
	wave := readFloat32Evidence(t, wavePath, "Gemma4 audio wave")
	audioPrompt, err := mustProjectorSession(t, runner).BuildAudioPrompt(
		context.Background(), language, wave, "", "What note do you hear? One word.",
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(audioPrompt.TokenIDs, audioGolden.InputIDs) {
		t.Fatalf("Gemma4 audio prompt IDs = %d, want %d", len(audioPrompt.TokenIDs), len(audioGolden.InputIDs))
	}
	audioToken := generatePromptFirstToken(t, language, audioPrompt)
	if len(audioGolden.Steps) == 0 || !slices.Contains(audioGolden.Steps[0].TopIDs, audioToken) {
		t.Fatalf("Gemma4 audio first token = %d, outside oracle top IDs", audioToken)
	}
	imageAudio, err := mustProjectorSession(t, runner).BuildMediaHistoryPrompt(
		context.Background(), language,
		[]projector.MediaInput{projector.NewImageMediaInput(imageSource), projector.NewAudioMediaInput(wave)},
		[]string{"Image: ", " Audio: ", " Compare them."},
	)
	if err != nil {
		t.Fatal(err)
	}
	audioImage, err := mustProjectorSession(t, runner).BuildMediaHistoryPrompt(
		context.Background(), language,
		[]projector.MediaInput{projector.NewAudioMediaInput(wave), projector.NewImageMediaInput(imageSource)},
		[]string{"Audio: ", " Image: ", " Compare them."},
	)
	if err != nil {
		t.Fatal(err)
	}
	imageElements, audioElements := len(imagePrompt.Embeddings), len(audioPrompt.Embeddings)
	if len(imageAudio.Embeddings) != imageElements+audioElements ||
		!slices.Equal(imageAudio.Embeddings[:imageElements], imagePrompt.Embeddings) ||
		!slices.Equal(imageAudio.Embeddings[imageElements:], audioPrompt.Embeddings) ||
		!slices.Equal(audioImage.Embeddings[:audioElements], audioPrompt.Embeddings) ||
		!slices.Equal(audioImage.Embeddings[audioElements:], imagePrompt.Embeddings) {
		t.Fatal("Gemma4 mixed-media embedding order differs")
	}
	imageTokens, audioTokens := len(imagePrompt.EmbeddingTokenIndices), len(audioPrompt.EmbeddingTokenIndices)
	if len(imageAudio.EmbeddingTokenIndices) != imageTokens+audioTokens ||
		len(audioImage.EmbeddingTokenIndices) != imageTokens+audioTokens ||
		len(imageAudio.AttentionBlocks) != 1 || len(audioImage.AttentionBlocks) != 1 ||
		imageAudio.AttentionBlocks[0].Start != imageAudio.EmbeddingTokenIndices[0] ||
		imageAudio.AttentionBlocks[0].End != imageAudio.EmbeddingTokenIndices[imageTokens-1]+1 ||
		audioImage.AttentionBlocks[0].Start != audioImage.EmbeddingTokenIndices[audioTokens] ||
		audioImage.AttentionBlocks[0].End != audioImage.EmbeddingTokenIndices[len(audioImage.EmbeddingTokenIndices)-1]+1 {
		t.Fatal("Gemma4 mixed-media token or attention order differs")
	}
	imageAudioToken := generatePromptFirstToken(t, language, imageAudio)
	audioImageToken := generatePromptFirstToken(t, language, audioImage)
	second := flipHorizontal(imageSource)
	forward, err := mustProjectorSession(t, runner).BuildVideoPrompt(
		context.Background(), language, []image.Image{imageSource, second}, "", "Describe the motion.", 2, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	reverse, err := mustProjectorSession(t, runner).BuildVideoPrompt(
		context.Background(), language, []image.Image{second, imageSource}, "", "Describe the motion.", 2, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(forward.AttentionBlocks) != 2 || !slices.Equal(forward.TokenIDs, reverse.TokenIDs) {
		t.Fatalf("Gemma4 video prompt contract = blocks %v/%v", forward.AttentionBlocks, reverse.AttentionBlocks)
	}
	chunk := len(forward.Embeddings) / 2
	if chunk == 0 || !slices.Equal(forward.Embeddings[:chunk], reverse.Embeddings[chunk:]) ||
		!slices.Equal(forward.Embeddings[chunk:], reverse.Embeddings[:chunk]) {
		t.Fatal("Gemma4 video frame order is not retained")
	}
	videoToken := generatePromptFirstToken(t, language, forward)
	t.Logf("Gemma4 12B real matrix: image exact token %d; audio top-set token %d; mixed ordered tokens %d/%d; video ordered token %d",
		imageToken, audioToken, imageAudioToken, audioImageToken, videoToken)
}

func mustProjectorSession(t *testing.T, source projector.Projector) projector.Session {
	t.Helper()
	session, err := projector.NewSession(source)
	if err != nil {
		t.Fatal(err)
	}
	return session
}

func generatePromptFirstToken(t *testing.T, runner *inference.Runner, prompt projector.MultimodalPrompt) tokenizer.TokenID {
	t.Helper()
	ids, projected, err := inference.ProjectedInputsForPrompt(runner, prompt)
	if err != nil {
		t.Fatal(err)
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0, TopK: 1, TopP: 1})
	if err != nil {
		t.Fatal(err)
	}
	generated, _, err := runner.Generate(context.Background(), "", inference.GenerateOptions{
		MaxNewTokens: 1, Sampler: greedy, PromptTokenIDs: ids, ProjectedInputs: &projected,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(generated) != len(ids)+1 {
		t.Fatalf("multimodal generation returned %d tokens, want %d", len(generated), len(ids)+1)
	}
	return generated[len(ids)]
}

func testQwen35ImageVideoParity(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Qwen3.5-4B-mmproj-bf16-facts-v2.gguf")
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Qwen3.5-4B-f16.gguf")
	imagePath := filepath.Join(roots.Models, "..", "fixtures", "qwen35_mm_image.png")
	imageGoldenPath := filepath.Join(roots.Models, "..", "fixtures", "qwen35_mm_golden.json")
	videoGoldenPath := filepath.Join(roots.Models, "..", "fixtures", "qwen35_video_golden.json")
	assertSHA256(t, projectorPath, "7b92fcb9bd3c70ef66bfdf6f9dc085ca8335cb85190ecd2eb6273aa1d057496e")
	assertSHA256(t, imagePath, "ccacbf46fc6dfe7545aed8a791f8ac343e1302ffbb7a1e899297afb68aa517b7")
	assertSHA256(t, imageGoldenPath, "774a9e8ffe2ea3ba37e142123ee35ddcf91c24baa43647b0a9af96fce74f8ea1")
	assertSHA256(t, videoGoldenPath, "38d61519eb39e98e577aeeebec6159be5efd6b2680e3b4c52d87da0047967505")
	var imageGolden qwen35VisionGolden
	var videoGolden qwen35VideoGolden
	decodeJSONEvidence(t, imageGoldenPath, &imageGolden)
	decodeJSONEvidence(t, videoGoldenPath, &videoGolden)
	imageFile, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	imageSource, decodeErr := png.Decode(imageFile)
	closeErr := imageFile.Close()
	if decodeErr != nil || closeErr != nil {
		t.Fatal(errors.Join(decodeErr, closeErr))
	}
	vision, err := projector.OpenAs[*projector.Qwen3VLRunner](context.Background(), projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatalf("UNAVAILABLE: Qwen3.5 projector or CUDA absent; parity NOT verified: %v", err)
	}
	defer vision.Close()
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionRequest, recipe.ResidencyDeviceF32,
	)
	if err != nil {
		t.Fatal(err)
	}
	language, err := inference.OpenWithProgram(context.Background(), &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer language.Close()
	processedImage, err := projector.PreprocessQwen3VLImage(imageSource, vision.Spec(), projector.DefaultQwen3VLPreprocessOptions())
	if err != nil {
		t.Fatal(err)
	}
	assertQwen35Grid(t, "image", processedImage, imageGolden.Grid)
	assertQwen35Probe(t, "image pixels", processedImage.PixelValues, imageGolden.Pixels, 0.15, 0)
	imagePrompt, err := mustProjectorSession(t, vision).BuildImagePrompt(
		context.Background(), language, imageSource, "", "What color dominates this image? One word.", true,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertQwen35Prompt(t, "image", imagePrompt, imageGolden.InputIDs, imageGolden.Tokens, imageGolden.Intermediates.Merger)
	assertQwen35FirstToken(t, language, imagePrompt, imageGolden.GeneratedTokenID)

	frames := qwen35GoldenFrames()
	processedVideo, err := projector.PreprocessQwen3VLFrames(frames, vision.Spec(), projector.DefaultQwen3VLVideoPreprocessOptions())
	if err != nil {
		t.Fatal(err)
	}
	assertQwen35Grid(t, "video", processedVideo, videoGolden.Grid)
	assertQwen35Probe(t, "video pixels", processedVideo.PixelValues, videoGolden.Pixels, 0.15, 0)
	videoPrompt, err := mustProjectorSession(t, vision).BuildVideoPrompt(
		context.Background(), language, frames, "", videoGolden.Question, 24, true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := projector.Qwen35VideoPromptText("", videoGolden.Question, processedVideo.GridT,
		(processedVideo.GridH/vision.Spec().MergeSize)*(processedVideo.GridW/vision.Spec().MergeSize), 24, true); got != videoGolden.PromptText {
		t.Fatal("Qwen3.5 video prompt text differs from Python oracle")
	}
	assertQwen35Prompt(t, "video", videoPrompt, videoGolden.InputIDs, videoGolden.Tokens, videoGolden.Intermediates.Merger)
	assertQwen35FirstToken(t, language, videoPrompt, videoGolden.GeneratedTokenID)
	t.Logf("Qwen3.5 real matrix: image %v/%d tokens; video %v/%d tokens; exact prompts, MRoPE, first tokens",
		imageGolden.Grid, imageGolden.Tokens, videoGolden.Grid, videoGolden.Tokens)
}

func decodeJSONEvidence(t *testing.T, path string, destination any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, destination); err != nil {
		t.Fatal(err)
	}
}

func assertQwen35Grid(t *testing.T, label string, input projector.Qwen3VLImage, want []int) {
	t.Helper()
	if len(want) != 3 || input.GridT != want[0] || input.GridH != want[1] || input.GridW != want[2] {
		t.Fatalf("Qwen3.5 %s grid = [%d %d %d], want %v", label, input.GridT, input.GridH, input.GridW, want)
	}
}

func assertQwen35Probe(t *testing.T, label string, values []float32, record qwen35Probe, absolute, l2Relative float64) {
	t.Helper()
	if len(record.ProbeIndex) != len(record.ProbeValue) {
		t.Fatalf("%s probe record is invalid", label)
	}
	for index, offset := range record.ProbeIndex {
		if offset < 0 || offset >= len(values) || math.Abs(float64(values[offset]-record.ProbeValue[index])) > absolute {
			t.Fatalf("%s probe %d differs", label, offset)
		}
	}
	if l2Relative <= 0 {
		return
	}
	var sum float64
	for _, value := range values {
		sum += float64(value) * float64(value)
	}
	if relative := math.Abs(math.Sqrt(sum)-record.L2) / record.L2; relative > l2Relative {
		t.Fatalf("%s L2 relative = %.6g, limit %.6g", label, relative, l2Relative)
	}
}

func assertQwen35Prompt(
	t *testing.T, label string, prompt projector.MultimodalPrompt,
	wantIDs []tokenizer.TokenID, wantTokens int, merger qwen35Probe,
) {
	t.Helper()
	if !slices.Equal(prompt.TokenIDs, wantIDs) || len(prompt.EmbeddingTokenIndices) != wantTokens {
		t.Fatalf("Qwen3.5 %s prompt = %d IDs/%d media, want %d/%d",
			label, len(prompt.TokenIDs), len(prompt.EmbeddingTokenIndices), len(wantIDs), wantTokens)
	}
	for axis := range prompt.MultiAxisPositions {
		if len(prompt.MultiAxisPositions[axis]) != len(prompt.TokenIDs) {
			t.Fatalf("Qwen3.5 %s MRoPE axis %d = %d, want %d",
				label, axis, len(prompt.MultiAxisPositions[axis]), len(prompt.TokenIDs))
		}
	}
	assertQwen35Probe(t, label+" merger", prompt.Embeddings, merger, 0.35, 0.03)
}

func assertQwen35FirstToken(t *testing.T, runner *inference.Runner, prompt projector.MultimodalPrompt, want []tokenizer.TokenID) {
	t.Helper()
	if len(want) == 0 {
		t.Fatal("Qwen3.5 generated-token oracle is empty")
	}
	ids, projected, err := inference.ProjectedInputsForPrompt(runner, prompt)
	if err != nil {
		t.Fatal(err)
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0, TopK: 1, TopP: 1})
	if err != nil {
		t.Fatal(err)
	}
	generated, _, err := runner.Generate(context.Background(), "", inference.GenerateOptions{
		MaxNewTokens: 1, Sampler: greedy, PromptTokenIDs: ids, ProjectedInputs: &projected,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(generated) != len(ids)+1 || generated[len(ids)] != want[0] {
		t.Fatalf("Qwen3.5 first token = %v, want %d", generated[len(ids):], want[0])
	}
}

func qwen35GoldenFrames() []image.Image {
	frames := make([]image.Image, 16)
	for temporal := range frames {
		frame := image.NewRGBA(image.Rect(0, 0, 224, 224))
		for y := range 224 {
			for x := range 224 {
				frame.SetRGBA(x, y, color.RGBA{
					R: uint8((x*4 + temporal*8) % 256), G: uint8((y*5 + temporal*4) % 256),
					B: uint8(((x+y)*3 + temporal*16) % 256), A: 255,
				})
			}
		}
		frames[temporal] = frame
	}
	return frames
}

func testGemmaE4BImageLanguageParity(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf")
	imagePath := testutil.FixturePath(t, "e4b_vision", "gemma4_mm_image.png")
	goldenPath := testutil.FixturePath(t, "e4b_vision", "image_language.json")
	assertSHA256(t, projectorPath, "185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e")
	assertSHA256(t, modelPath, "cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501")
	assertSHA256(t, imagePath, "996fea5cea787f3abb6d1377fc88642ade6bdb8dc9a7b9e6ad909e58687b776e")
	assertSHA256(t, goldenPath, "49b5624323fc4e2e87a2673d9b8ab8880d285e1bcb736dad7dcbea6ecd919232")
	var golden e4bImageLanguageGolden
	goldenRaw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: E4B image-language golden absent; parity NOT verified: %v", err)
	}
	if err := json.Unmarshal(goldenRaw, &golden); err != nil {
		t.Fatal(err)
	}
	imageFile, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	source, decodeErr := png.Decode(imageFile)
	closeErr := imageFile.Close()
	if decodeErr != nil || closeErr != nil {
		t.Fatal(errors.Join(decodeErr, closeErr))
	}
	projectorRunner, err := projector.OpenAs[*projector.Gemma4TowerRunner](context.Background(), projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatalf("UNAVAILABLE: E4B projector absent or CUDA unavailable; parity NOT verified: %v", err)
	}
	input, err := projector.PreprocessVisionTowerImage(source, projectorRunner.Spec().Vision)
	if err != nil {
		_ = projectorRunner.Close()
		t.Fatal(err)
	}
	projected, err := projectorRunner.EncodeVisionPatches(context.Background(), input.PixelValues, input.Positions)
	projectorCloseErr := projectorRunner.Close()
	if err != nil || projectorCloseErr != nil {
		t.Fatal(errors.Join(err, projectorCloseErr))
	}
	if projected.SoftTokens != golden.ImageCount {
		t.Fatalf("E4B image soft tokens = %d, want %d", projected.SoftTokens, golden.ImageCount)
	}
	width := int(projected.Embeddings.Shape.Dims[0])
	overrides := make([]inference.EmbeddingOverride, projected.SoftTokens)
	for index := range overrides {
		start := index * width
		overrides[index] = inference.EmbeddingOverride{
			TokenIndex: uint32(len(golden.Prefix) + index),
			Embedding:  slices.Clone(projected.Embeddings.Data[start : start+width]),
		}
	}
	promptIDs := slices.Clone(golden.Prefix)
	promptIDs = append(promptIDs, slices.Repeat([]tokenizer.TokenID{golden.ImageToken}, golden.ImageCount)...)
	promptIDs = append(promptIDs, golden.Suffix...)
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionRequest, recipe.ResidencyHybridNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	languageRunner, err := inference.OpenWithProgram(context.Background(), &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer languageRunner.Close()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	var evaluation inference.PromptEvaluation
	ids, _, err := languageRunner.Generate(context.Background(), "", inference.GenerateOptions{
		MaxNewTokens:   1,
		Sampler:        greedy,
		PromptTokenIDs: promptIDs,
		ProjectedInputs: &inference.ProjectedInputs{
			EmbeddingOverrides: overrides,
			BidirectionalAttentionBlocks: []inference.AttentionBlock{{
				Start: uint32(len(golden.Prefix)),
				End:   uint32(len(golden.Prefix) + golden.ImageCount),
			}},
		},
		OnPromptEvaluated: func(got inference.PromptEvaluation) { evaluation = got },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != len(promptIDs)+1 || ids[len(ids)-1] != golden.NextToken {
		t.Fatalf("E4B image-language next token = %v, want %d (%s)", ids[len(promptIDs):], golden.NextToken, golden.NextPiece)
	}
	t.Logf("E4B image-language parity: %d prompt + %d projected tokens -> %d (%s); prefill %s",
		len(promptIDs), len(overrides), golden.NextToken, golden.NextPiece, evaluation.Duration)
}

func testGemmaE4BAudioParity(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	featurePath := testutil.FixturePath(t, "e4b_audio", "input_features.f32")
	goldenPath := testutil.FixturePath(t, "e4b_audio", "g_audio_f32.json")
	wavePath := testutil.FixturePath(t, "e4b_audio", "wave.f32")
	assertSHA256(t, projectorPath, "185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e")
	assertSHA256(t, featurePath, "6faf97d1bf73ab3631f38332bf0539d43ebd0eddaf76865728288032dc939fca")
	assertSHA256(t, goldenPath, "9de625447fbc7ab1f12d2de6c73700aa2daa9e476f3e4790e6632a4623371bbb")
	assertSHA256(t, wavePath, "98c8d1a25f96bbdfff5ca6153c9b18fad9740e300008618a970c18384b35d404")
	goldenRaw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("UNAVAILABLE: E4B audio golden absent; parity NOT verified: %v", err)
	}
	var golden e4bAudioGolden
	if err := json.Unmarshal(goldenRaw, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.InputShape) != 3 || golden.InputShape[0] != 1 {
		t.Fatalf("E4B audio golden input shape = %v", golden.InputShape)
	}
	features := readFloat32Evidence(t, featurePath, "E4B audio features")
	wave := readFloat32Evidence(t, wavePath, "E4B audio wave")
	runner, err := projector.OpenAs[*projector.Gemma4TowerRunner](context.Background(), projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatalf("UNAVAILABLE: E4B projector absent or CUDA unavailable; parity NOT verified: %v", err)
	}
	defer runner.Close()
	profile, err := projector.NewAudioProjectionProfile(10000)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	output, trace, err := runner.EncodeAudioTrace(
		context.Background(), wave, runner.Spec().Audio.SampleRate, profile,
	)
	if err != nil {
		t.Fatal(err)
	}
	frontend, ok := trace.Stages["frontend"]
	if !ok {
		t.Fatal("E4B audio trace lacks frontend")
	}
	featureError := 0.0
	if int(frontend.Shape.Dims[tensor.SingletonExtent]) != golden.InputShape[1] || len(frontend.Data) != len(features) {
		t.Fatalf("E4B audio frontend shape = %v, want [%d,%d]",
			frontend.Shape, golden.InputShape[1], len(features))
	}
	for index, value := range frontend.Data {
		featureError = max(featureError, math.Abs(float64(value-features[index])))
	}
	if featureError > 1e-2 {
		t.Fatalf("E4B audio frontend max error = %.6g, adaptive limit 0.01", featureError)
	}
	wall := time.Since(start)
	if output.SoftTokens != golden.SoftCount || int(output.Embeddings.Shape.Dims[0]) != golden.SoftDim {
		t.Fatalf("E4B audio shape = [%d,%d], want [%d,%d]",
			output.SoftTokens, output.Embeddings.Shape.Dims[0], golden.SoftCount, golden.SoftDim)
	}
	worstName, worstRelative := "", 0.0
	stageNames := []string{"subsample"}
	for layer := range runner.Spec().Audio.Layers {
		stageNames = append(stageNames, fmt.Sprintf("enc%d", layer))
	}
	for _, name := range stageNames {
		got, ok := trace.Stages[name]
		if !ok {
			t.Fatalf("E4B audio trace lacks %s", name)
		}
		relative := sampledRelative(got.Data, golden.Layers[name], int(got.Shape.Dims[0]))
		if relative > worstRelative {
			worstName, worstRelative = name, relative
		}
		if relative > 5e-3 {
			t.Errorf("E4B audio %s relative = %.6g, limit 0.005", name, relative)
		}
	}
	soft := trace.Stages["soft_tokens"]
	softRelative := sampledRelative(soft.Data, golden.SoftTokens, int(soft.Shape.Dims[0]))
	if softRelative > 5e-3 {
		t.Fatalf("E4B audio soft-token relative = %.6g, limit 0.005", softRelative)
	}
	t.Logf("E4B audio parity: %d samples -> %d feature frames -> %d x %d soft tokens in %s; frontend %.6g; worst stage %s %.6g; soft %.6g",
		len(wave), golden.InputShape[1], output.SoftTokens, output.Embeddings.Shape.Dims[0], wall,
		featureError, worstName, worstRelative, softRelative)
}

func readFloat32Evidence(t *testing.T, path, label string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("UNAVAILABLE: %s absent; parity NOT verified: %v", label, err)
	}
	if len(raw)%4 != 0 {
		t.Fatalf("%s storage is invalid", label)
	}
	values := make([]float32, len(raw)/4)
	if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, values); err != nil {
		t.Fatalf("read %s: %v", label, err)
	}
	return values
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
	assertSHA256(t, projectorPath, "185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e")
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
	runner, err := projector.OpenAs[*projector.Gemma4TowerRunner](context.Background(), projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatalf("UNAVAILABLE: E4B projector absent or CUDA unavailable; parity NOT verified: %v", err)
	}
	defer func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	}()
	input, err := projector.PreprocessVisionTowerImage(source, runner.Spec().Vision)
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

func testGemmaE4BResizeParity(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	imagePath := testutil.FixturePath(t, "e4b_vision", "gemma4_mm_resize_image.png")
	goldenPath := testutil.FixturePath(t, "e4b_vision", "gemma4_mm_resize_golden.json")
	assertSHA256(t, imagePath, "f7ee41858ae2238bbd2cb41220852c9316c08d4f8b7d027c7da4e0308f3d5274")
	assertSHA256(t, goldenPath, "265db1d122923064cb36484bd73285a231277a7d7a8985c442c4a87957fa9335")
	var golden struct {
		ResizeOutHW   []int `json:"resize_out_hw"`
		NumSoftTokens int   `json:"num_soft_tokens"`
		ResizedUint8  struct {
			ProbeIndex []int     `json:"probe_index"`
			ProbeValue []float64 `json:"probe_value"`
		} `json:"resized_uint8"`
	}
	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	source, decodeErr := png.Decode(file)
	closeErr := file.Close()
	if decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	runner, err := projector.OpenAs[*projector.Gemma4TowerRunner](context.Background(), projectorPath, projector.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	input, err := projector.PreprocessVisionTowerImage(source, runner.Spec().Vision)
	if err != nil {
		t.Fatal(err)
	}
	if len(golden.ResizeOutHW) != 2 || input.GridH*runner.Spec().Vision.PatchSize != golden.ResizeOutHW[0] ||
		input.GridW*runner.Spec().Vision.PatchSize != golden.ResizeOutHW[1] ||
		len(input.PixelValues)/(runner.Spec().Vision.PatchSize*runner.Spec().Vision.PatchSize*3)/9 != golden.NumSoftTokens {
		t.Fatalf("E4B resize grid=%dx%d, soft=%d; want %v, %d",
			input.GridH, input.GridW, len(input.PixelValues)/(runner.Spec().Vision.PatchSize*runner.Spec().Vision.PatchSize*3)/9,
			golden.ResizeOutHW, golden.NumSoftTokens)
	}
	chw := gemmaE4BInputCHW(input, runner.Spec().Vision.PatchSize)
	worst := 0.0
	for index, offset := range golden.ResizedUint8.ProbeIndex {
		worst = max(worst, math.Abs(float64(chw[offset])*255-golden.ResizedUint8.ProbeValue[index]))
	}
	if worst > 1.01 {
		t.Fatalf("E4B resized uint8 probe error = %.6g, limit 1.01", worst)
	}
	t.Logf("E4B dynamic resize parity: %dx%d, %d soft tokens, uint8 probe max %.6g",
		golden.ResizeOutHW[1], golden.ResizeOutHW[0], golden.NumSoftTokens, worst)
}

func gemmaE4BInputCHW(input projector.Gemma4VisionTowerInput, patchSize int) []float32 {
	height, width := input.GridH*patchSize, input.GridW*patchSize
	result := make([]float32, 3*height*width)
	patchWidth := patchSize * patchSize * 3
	for patchY := range input.GridH {
		for patchX := range input.GridW {
			row := patchY*input.GridW + patchX
			for y := range patchSize {
				for x := range patchSize {
					pixel := row*patchWidth + (y*patchSize+x)*3
					global := (patchY*patchSize+y)*width + patchX*patchSize + x
					for channel := range 3 {
						result[channel*height*width+global] = input.PixelValues[pixel+channel]
					}
				}
			}
		}
	}
	return result
}

func testGemmaE4BVideoOrder(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	imagePath := testutil.FixturePath(t, "e4b_vision", "gemma4_mm_image.png")
	file, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	first, decodeErr := png.Decode(file)
	closeErr := file.Close()
	if decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	second := flipHorizontal(first)
	runner, err := projector.OpenAs[*projector.Gemma4TowerRunner](context.Background(), projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	forward, err := runner.EncodeVisionFrames(context.Background(), []image.Image{first, second})
	if err != nil {
		t.Fatal(err)
	}
	reverse, err := runner.EncodeVisionFrames(context.Background(), []image.Image{second, first})
	if err != nil {
		t.Fatal(err)
	}
	if forward.Frames != 2 || forward.TokensPerFrame != runner.Spec().Vision.MaxVideoTokens ||
		reverse.TokensPerFrame != forward.TokensPerFrame {
		t.Fatalf("E4B video contract forward=%+v reverse=%+v", forward, reverse)
	}
	chunk := forward.TokensPerFrame * runner.Spec().Vision.ProjectionDim
	if slices.Equal(forward.Embeddings.Data[:chunk], forward.Embeddings.Data[chunk:]) {
		t.Fatal("E4B video frames collapsed to identical embeddings")
	}
	if !slices.Equal(forward.Embeddings.Data[:chunk], reverse.Embeddings.Data[chunk:]) ||
		!slices.Equal(forward.Embeddings.Data[chunk:], reverse.Embeddings.Data[:chunk]) {
		t.Fatal("E4B video frame order is not preserved exactly")
	}
	t.Logf("E4B video order: 2 frames x %d soft tokens; exact frame-major reversal", forward.TokensPerFrame)
}

func flipHorizontal(source image.Image) image.Image {
	bounds := source.Bounds()
	result := image.NewRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	for y := range bounds.Dy() {
		for x := range bounds.Dx() {
			result.Set(x, y, source.At(bounds.Min.X+bounds.Dx()-1-x, bounds.Min.Y+y))
		}
	}
	return result
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
