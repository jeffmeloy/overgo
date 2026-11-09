package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestARenderKeepsItsClipsInOneFormat holds a clip render to one format: a directory of clips drawn before a render
// named its format holds GIFs, so a render that writes MP4s draws into another directory and one that asks for GIFs
// carries on, and a picture render names none. A pair screen whose sides were written in different formats is refused
// before any output is embedded, since a GIF lies apart from an MP4 by its palette as well as its model.
func TestARenderKeepsItsClipsInOneFormat(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "clip-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "clip-recipe")
	held := renderManifest{Model: model, Recipe: recipeID, Task: recipe.TaskVideoGen, CodeCommit: strings.Repeat("a", 40), Draws: map[string]renderedMedia{}}
	directory := t.TempDir()
	testcheck.NoError(t, jsonfile.Write(filepath.Join(directory, renderManifestFile), held, renderFileMode))
	mp4, gif := held, held
	mp4.Format, gif.Format = "mp4", gifFormat
	if _, err := alreadyDrawn(directory, mp4); err == nil {
		t.Fatal("an MP4 render was drawn into a directory of GIFs")
	}
	if _, err := alreadyDrawn(directory, gif); err != nil {
		t.Fatalf("a GIF render of the same recipe was refused its directory: %v", err)
	}
	if format := (renderManifest{Task: recipe.TaskImageGen}).clipFormat(); format != "" {
		t.Fatalf("a picture render names the clip format %q", format)
	}

	side := func(format string) string {
		directory := t.TempDir()
		file := "clip" + drawnExtensions["video/mp4"]
		if format != "mp4" {
			file = "clip" + drawnExtensions["image/gif"]
		}
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, file), []byte("clip"), renderFileMode))
		manifest := map[string]any{"task": recipe.TaskVideoGen, "format": format, "draws": map[string]any{
			"key": map[string]any{"file": file, "request": map[string]any{"prompt": "a fox", "seed": 1}},
		}}
		testcheck.NoError(t, jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, renderFileMode))
		return directory
	}
	if _, err := screenPaths(t.Context(), nil, side("mp4"), side(""), side("")); err == nil || !strings.Contains(err.Error(), "one format") {
		t.Fatalf("a screen of an MP4 candidate beside GIF references gave %v", err)
	}

	// A rendered suite names each clip by its request's key, which the render's manifest binds to the clip's file; a
	// folder of clips with no manifest (VidGen's) names each by its file.
	rendered := side("mp4")
	if file, err := clipFile(rendered, "key"); err != nil || filepath.Base(file) != "clip.mp4" {
		t.Fatalf("the rendered key resolves to %q (%v), want its clip", file, err)
	}
	plain := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(plain, "source.mp4"), []byte("clip"), renderFileMode))
	if file, err := clipFile(plain, "source.mp4"); err != nil || filepath.Base(file) != "source.mp4" {
		t.Fatalf("a clip of a plain folder resolves to %q (%v)", file, err)
	}
}
