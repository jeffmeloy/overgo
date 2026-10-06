package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// renderedDirectory writes a picture named by its digest and the manifest that
// binds the directory to a model and recipe.
func renderedDirectory(t *testing.T, model, recipeID artifact.ID) (directory, name string) {
	t.Helper()
	directory = t.TempDir()
	picture := []byte("a picture the model drew")
	sum := sha256.Sum256(picture)
	name = hex.EncodeToString(sum[:]) + ".png"
	if err := os.WriteFile(filepath.Join(directory, name), picture, 0o600); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(renderManifest{Model: model, Recipe: recipeID, Task: recipe.TaskImageGen, CodeCommit: "abc", Files: map[string]string{name: "coco_1"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, renderManifestFile), data, 0o600); err != nil {
		t.Fatal(err)
	}
	return directory, name
}

// TestAPictureIsJudgedOnlyWhenTheEvaluatedRecipeDrewIt holds the render
// manifest to what binds a picture to the model being scored: the model and
// recipe it names, a picture it lists, and bytes that are the digest the file
// is named for.
func TestAPictureIsJudgedOnlyWhenTheEvaluatedRecipeDrewIt(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "drawing-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "drawing-recipe")
	directory, name := renderedDirectory(t, model, recipeID)
	if err := verifyRendered(directory, model, recipeID, name); err != nil {
		t.Fatalf("a picture the recipe drew was refused: %v", err)
	}
	if err := verifyRendered(directory, testutil.ArtifactID(t, artifact.KindModel, "another-model"), recipeID, name); err == nil {
		t.Error("a picture another model drew was judged as this one's")
	}
	if err := verifyRendered(directory, model, testutil.ArtifactID(t, artifact.KindRecipe, "another-recipe"), name); err == nil {
		t.Error("a picture another recipe drew was judged as this one's")
	}
	if err := verifyRendered(directory, model, recipeID, "unlisted.png"); err == nil {
		t.Error("a picture the manifest does not list was judged")
	}
	if err := os.WriteFile(filepath.Join(directory, name), []byte("another picture"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := verifyRendered(directory, model, recipeID, name); err == nil {
		t.Error("bytes that are not the digest the file is named for were judged")
	}
	if err := verifyRendered(t.TempDir(), model, recipeID, name); err == nil {
		t.Error("a directory with no render manifest was judged")
	}
}

// TestAJudgedRuntimeAnswersOnlyAboutAVerifiedPicture holds the judge to its
// contract: it names itself, it asks nothing of a prompt that names no picture,
// and it looks at nothing the manifest does not bind to the evaluated recipe.
func TestAJudgedRuntimeAnswersOnlyAboutAVerifiedPicture(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "drawing-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "drawing-recipe")
	directory, name := renderedDirectory(t, model, recipeID)
	judge := judgedRuntime{projectionRuntime: projectionRuntime{root: directory}, judge: "vision-judge", model: model, recipe: recipeID}
	if judge.Judge() != "vision-judge" {
		t.Fatalf("the judge names itself %q", judge.Judge())
	}
	if _, _, err := judge.Generate(t.Context(), "Which?", inference.GenerateOptions{}); err == nil {
		t.Error("a prompt that names no picture was answered")
	}
	foreign := judge
	foreign.recipe = testutil.ArtifactID(t, artifact.KindRecipe, "another-recipe")
	if _, _, err := foreign.Generate(t.Context(), "<overgo-media kind=image name="+name+">\nWhich?", inference.GenerateOptions{}); err == nil {
		t.Error("a picture another recipe drew was answered about")
	}
}
