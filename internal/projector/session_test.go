package projector

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"image"
	"reflect"
	"testing"
)

func TestCompiledSessionOwnsNoPromptWrappers(t *testing.T) {
	packages, err := parser.ParseDir(token.NewFileSet(), ".", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	forbidden := map[string]struct{}{
		"imagesPrompt": {}, "videoPrompt": {}, "audioPrompt": {},
		"audioSampleRate": {}, "mediaHistoryPrompt": {},
		"imagePromptProgram": {}, "mediaPromptProgram": {},
	}
	for _, file := range packages["projector"].Files {
		for _, declaration := range file.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || function.Recv == nil {
				continue
			}
			if _, legacy := forbidden[function.Name.Name]; legacy {
				t.Errorf("legacy prompt wrapper remains: %s", function.Name.Name)
			}
		}
	}
}

func TestPromptRecipeProgramsCoverCatalog(t *testing.T) {
	seen := make(map[string]struct{}, len(projectorCatalog))
	for _, descriptor := range projectorCatalog {
		if _, duplicate := seen[descriptor.kind]; duplicate {
			t.Fatalf("duplicate projector recipe %q", descriptor.kind)
		}
		seen[descriptor.kind] = struct{}{}
		if len(descriptor.media) > 0 && descriptor.prompt == nil {
			t.Fatalf("media projector %q has no prompt recipe", descriptor.kind)
		}
	}
}

// TestCompiledSessionOwnsPromptDispatch checks session-only prompt entry points.
func TestCompiledSessionOwnsPromptDispatch(t *testing.T) {
	t.Run("nil source", func(t *testing.T) {
		if _, err := NewSession(nil); err == nil {
			t.Fatal("nil session source accepted")
		}
	})
	t.Run("cogvlm image prompt", func(t *testing.T) {
		runner, err := openImageProjectorAs[*CogVLMVisionRunner](writeTinyCogVLM(t, false), OpenOptions{})
		if err != nil {
			t.Fatal(err)
		}
		defer runner.Close()
		session := testSession(t, runner)
		tok := &cogVLMPromptTokenizer{}
		single, err := session.BuildImagePrompt(
			context.Background(), tok, image.NewRGBA(image.Rect(0, 0, 4, 4)), "before", "after", false,
		)
		if err != nil {
			t.Fatal(err)
		}
		if tok.text != "Question: beforeafter Answer:" {
			t.Fatalf("session image prompt text = %q", tok.text)
		}
		multi, err := session.BuildImagesPrompt(
			context.Background(), tok, []image.Image{image.NewRGBA(image.Rect(0, 0, 4, 4))},
			[]string{"before", "after"}, PromptOptions{},
		)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(single, multi) {
			t.Fatal("single-image session prompt differs from one-element images prompt")
		}
		if _, err := session.BuildVideoPrompt(
			context.Background(), tok, []image.Image{image.NewRGBA(image.Rect(0, 0, 4, 4))}, "", "q", 1, false,
		); err == nil {
			t.Fatal("video prompt accepted by an image-only session")
		}
		if _, err := session.BuildAudioPrompt(context.Background(), tok, []float32{0}, "", "q"); err == nil {
			t.Fatal("audio prompt accepted by an image-only session")
		}
		if _, err := session.BuildMediaHistoryPrompt(
			context.Background(), tok, []MediaInput{NewImageMediaInput(image.NewRGBA(image.Rect(0, 0, 4, 4)))}, []string{"a", "b"},
		); err == nil {
			t.Fatal("media history prompt accepted by an image-only session")
		}
	})
	t.Run("mimovl images prompt", func(t *testing.T) {
		runner, err := openImageProjectorAs[*MiMoVLRunner](writeTinyMiMoVL(t, tinyMiMoVLTensors(false)), OpenOptions{})
		if err != nil {
			t.Fatal(err)
		}
		defer runner.Close()
		tok := &mimoVLPromptTokenizer{}
		prompt, err := testSession(t, runner).BuildImagesPrompt(
			context.Background(), tok,
			[]image.Image{image.NewRGBA(image.Rect(0, 0, 4, 4)), image.NewRGBA(image.Rect(0, 0, 4, 4))},
			[]string{"a", "b", "c"}, PromptOptions{},
		)
		if err != nil {
			t.Fatal(err)
		}
		if prompt.EmbeddingWidth != 4 || len(prompt.EmbeddingTokenIndices) != 8 {
			t.Fatalf("session images prompt = width %d indices %v", prompt.EmbeddingWidth, prompt.EmbeddingTokenIndices)
		}
	})
}
