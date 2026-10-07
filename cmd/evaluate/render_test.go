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

// TestARenderReusesOnlyWhatTheSameRecipeAndSettingsDrew holds what a render may take
// from a directory already drawn into: the files of the captions its manifest lists,
// when the model, recipe, task and settings are the render's own, with the commit that
// drew them kept; nothing from a directory with no manifest; and a refusal for media
// another recipe or other settings drew.
func TestARenderReusesOnlyWhatTheSameRecipeAndSettingsDrew(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "drawing-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "drawing-recipe")
	want := renderManifest{Model: model, Recipe: recipeID, Task: recipe.TaskImageGen, CodeCommit: "new", Files: map[string]string{}}
	fresh, drawn, err := alreadyDrawn(t.TempDir(), want)
	if err != nil || len(drawn) != 0 || fresh.CodeCommit != "new" {
		t.Fatalf("an empty directory gave %+v, %v, %v; want nothing to reuse and this render's commit", fresh, drawn, err)
	}
	directory, name := renderedDirectory(t, model, recipeID)
	held, drawn, err := alreadyDrawn(directory, want)
	if err != nil || drawn["coco_1"] != name || held.CodeCommit != "abc" {
		t.Fatalf("a directory this recipe drew gave %+v, %v, %v; want its picture and the commit that drew it", held, drawn, err)
	}
	other := want
	other.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "another-recipe")
	if _, _, err := alreadyDrawn(directory, other); err == nil {
		t.Error("media another recipe drew were reused")
	}
	resized := want
	resized.Settings = map[string]float64{"width": 416}
	if _, _, err := alreadyDrawn(directory, resized); err == nil {
		t.Error("media drawn with other settings were reused")
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
