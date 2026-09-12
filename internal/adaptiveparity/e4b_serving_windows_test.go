package adaptiveparity

import (
	"image"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestE4BServingModalities checks the real compiled session instead of bypassing
// serving through the numerical tower methods.
func TestE4BServingModalities(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv("OVERGO_CUDA_TEST") != "1" {
		t.Skip("set OVERGO_CUDA_TEST=1 for real E4B serving modality validation")
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	assertSHA256(t, path, "185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e")
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf")
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	candidate, err := modelintake.PrepareProjectionCandidate(t.Context(), store, modelPath, path)
	if err != nil {
		t.Fatal(err)
	}
	session, err := projector.OpenSession(t.Context(), path, projector.OpenOptions{CUDA: true, MediaPreprocess: candidate.Processor})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	want := projector.SessionCapabilities{Image: true, MultiImage: true, Audio: true, Video: true, MediaHistory: true}
	if got := session.Capabilities(); got != want {
		t.Fatalf("real E4B serving modalities = %+v, want %+v; tower parity alone cannot admit serving", got, want)
	}
	assertSHA256(t, modelPath, "cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501")
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionRequest, recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	language, err := inference.OpenWithProgram(t.Context(), &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer language.Close()
	imagePath := testutil.FixturePath(t, "e4b_vision", "gemma4_mm_image.png")
	assertSHA256(t, imagePath, "996fea5cea787f3abb6d1377fc88642ade6bdb8dc9a7b9e6ad909e58687b776e")
	file, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	picture, err := png.Decode(file)
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("image: %v; close: %v", err, closeErr)
	}
	goldenPath := testutil.FixturePath(t, "e4b_vision", "image_language.json")
	assertSHA256(t, goldenPath, "49b5624323fc4e2e87a2673d9b8ab8880d285e1bcb736dad7dcbea6ecd919232")
	var golden e4bImageLanguageGolden
	decodeJSONEvidence(t, goldenPath, &golden)
	var imagePrompt projector.MultimodalPrompt
	t.Run("image-language", func(t *testing.T) {
		imagePrompt, err = session.BuildImagePrompt(t.Context(), language, picture, "", "What is in this image? One word.", false)
		if err != nil {
			t.Fatal(err)
		}
		wantIDs := append(slices.Clone(golden.Prefix), slices.Repeat([]tokenizer.TokenID{golden.ImageToken}, golden.ImageCount)...)
		wantIDs = append(wantIDs, golden.Suffix...)
		if !slices.Equal(imagePrompt.TokenIDs, wantIDs) {
			t.Fatal("serving image prompt differs from the independent E4B oracle")
		}
		if got := generatePromptFirstToken(t, language, imagePrompt); got != golden.NextToken {
			t.Fatalf("image token=%d, oracle=%d", got, golden.NextToken)
		}
	})
	wavePath := testutil.FixturePath(t, "e4b_audio", "wave.f32")
	assertSHA256(t, wavePath, "98c8d1a25f96bbdfff5ca6153c9b18fad9740e300008618a970c18384b35d404")
	wave := readFloat32Evidence(t, wavePath, "E4B audio wave")
	var audioPrompt projector.MultimodalPrompt
	t.Run("audio-language", func(t *testing.T) {
		audioPrompt, err = session.BuildAudioPrompt(t.Context(), language, wave, "", "What note do you hear? One word.")
		if err != nil {
			t.Fatal(err)
		}
		var golden e4bAudioGolden
		goldenPath := testutil.FixturePath(t, "e4b_audio", "g_audio_f32.json")
		assertSHA256(t, goldenPath, "9de625447fbc7ab1f12d2de6c73700aa2daa9e476f3e4790e6632a4623371bbb")
		decodeJSONEvidence(t, goldenPath, &golden)
		if len(audioPrompt.EmbeddingTokenIndices) != golden.SoftCount || audioPrompt.EmbeddingWidth != golden.SoftDim {
			t.Fatal("serving audio projection differs from oracle geometry")
		}
		t.Logf("audio compiled through language; first token=%d; transcription quality remains a separate dataset assertion", generatePromptFirstToken(t, language, audioPrompt))
	})
	t.Run("video-language", func(t *testing.T) {
		prompt, err := session.BuildVideoPrompt(t.Context(), language, []image.Image{picture, picture}, "", "Describe the frames.", 2, false)
		if err != nil {
			t.Fatal(err)
		}
		// This pinned artifact declares causal image attention. Frame presence
		// is independent of bidirectional blocks, which would change its mask.
		if candidate.Config == nil || candidate.Config.Generation == nil || candidate.Config.Generation.ImageAttention != "causal" {
			t.Fatal("pinned E4B causal media policy is unavailable")
		}
		if len(prompt.AttentionBlocks) != 0 {
			t.Fatal("causal video acquired bidirectional attention blocks")
		}
		frames := 0
		for index, token := range prompt.EmbeddingTokenIndices {
			if index == 0 || token != prompt.EmbeddingTokenIndices[index-1]+1 {
				frames++
			}
		}
		if frames != 2 || len(prompt.Embeddings) != len(prompt.EmbeddingTokenIndices)*prompt.EmbeddingWidth {
			t.Fatal("video frame embeddings lost")
		}
		t.Logf("video compiled through language; first token=%d; task quality remains a separate dataset assertion", generatePromptFirstToken(t, language, prompt))
	})
	if t.Failed() {
		return
	}
	t.Run("ordered-mixed-history", func(t *testing.T) {
		prompt, err := session.BuildMediaHistoryPrompt(t.Context(), language, []projector.MediaInput{projector.NewImageMediaInput(picture), projector.NewAudioMediaInput(wave)}, []string{"<bos><|turn>user\n", "\n", "Describe the inputs.<turn|>\n<|turn>model\n"})
		if err != nil {
			t.Fatal(err)
		}
		wantEmbeddings := append(slices.Clone(imagePrompt.Embeddings), audioPrompt.Embeddings...)
		if !slices.Equal(prompt.Embeddings, wantEmbeddings) {
			t.Fatal("mixed-history serving reordered or changed projected inputs")
		}
		t.Logf("ordered mixed history compiled through language; first token=%d", generatePromptFirstToken(t, language, prompt))
	})
	t.Log("serving producer: image exact prompt/token oracle; audio/video/mixed projection-to-language execution; HTTP routes, full task outputs, dataset quality and resource soak are NOT accepted here")
}
