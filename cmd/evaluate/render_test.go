package main

import (
	"strings"

	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// renderedDirectory writes a picture named by its digest and the manifest that
// binds the directory to a model and recipe.
func renderedDirectory(t *testing.T, model, recipeID artifact.ID) (directory, key string, store *overgodb.Store) {
	t.Helper()
	directory = t.TempDir()
	var err error
	store, err = overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	picture := []byte("a picture the model drew")
	output, err := artifact.IdentifyBytes(artifact.KindOutput, picture)
	if err != nil {
		t.Fatal(err)
	}
	request, key, err := renderRequest(map[string]any{"seed": float64(1)}, nil, renderPrompt{ID: "coco_1", Prompt: "a cat"}, 0)
	if err != nil {
		t.Fatal(err)
	}
	inputContent, err := capabilityruntime.InputContent("overgo.image-gen-input.v1", json.RawMessage(request))
	if err != nil {
		t.Fatal(err)
	}
	input := inputContent.Descriptor.ID
	commit := strings.Repeat("a", 40)
	run, err := runrecord.NewBoundRun(recipeID, runrecord.OutcomeSucceeded, []artifact.ID{input}, []artifact.ID{output}, "", commit, testutil.ArtifactID(t, artifact.KindEvidence, "environment"), 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	content, err := run.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(t.Context(), store, artifact.Batch{Key: "render-fixture", Contents: []artifact.Content{content, inputContent}}); err != nil {
		t.Fatal(err)
	}
	name := output.DigestHex() + ".png"
	if err := os.WriteFile(filepath.Join(directory, name), picture, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest := renderManifest{Model: model, Recipe: recipeID, Task: recipe.TaskImageGen, CodeCommit: commit, Draws: map[string]renderedMedia{
		key: {File: name, Request: request, Input: input, Output: output, Run: run.ID},
	}}
	if err := jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	return directory, key, store
}

// TestAPictureIsJudgedOnlyWhenTheEvaluatedRecipeDrewIt holds the render
// manifest to what binds a picture to the model being scored: the model and
// recipe it names, a picture it lists, and bytes that are the digest the file
// is named for.
func TestAPictureIsJudgedOnlyWhenTheEvaluatedRecipeDrewIt(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "drawing-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "drawing-recipe")
	directory, name, store := renderedDirectory(t, model, recipeID)
	held, err := verifyRendered(t.Context(), store, directory, model, recipeID, name)
	if err != nil {
		t.Fatal(err)
	}
	var manifest renderManifest
	if err := jsonfile.Decode(filepath.Join(directory, renderManifestFile), &manifest); err != nil {
		t.Fatal(err)
	}
	for label, change := range map[string]func(*renderedMedia){
		"missing producer":     func(v *renderedMedia) { v.Run = artifact.ID{} },
		"unrecorded producer":  func(v *renderedMedia) { v.Run = testutil.ArtifactID(t, artifact.KindRun, "other") },
		"changed request":      func(v *renderedMedia) { v.Request = json.RawMessage(`{"prompt":"another"}`) },
		"wrong producer input": func(v *renderedMedia) { v.Input = testutil.ArtifactID(t, artifact.KindFile, "other") },
	} {
		changed := held
		change(&changed)
		manifest.Draws[name] = changed
		if err := jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := verifyRendered(t.Context(), store, directory, model, recipeID, name); err == nil {
			t.Errorf("%s was qualified", label)
		}
	}
	for label, raw := range map[string]json.RawMessage{
		"changed prompt": json.RawMessage(`{"prompt":"not what the producer consumed","seed":1}`),
		"missing prompt": json.RawMessage(`{"seed":1}`),
	} {
		forged := held
		forged.Request = raw
		forgedKey, err := artifact.JSONID(artifact.KindFile, forged.Request)
		if err != nil {
			t.Fatal(err)
		}
		manifest.Draws = map[string]renderedMedia{forgedKey.DigestHex(): forged}
		if err := jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := verifyRendered(t.Context(), store, directory, model, recipeID, forgedKey.DigestHex()); err == nil {
			t.Errorf("%s borrowed another producer", label)
		}
	}
	zero := held
	var zeroFields map[string]any
	if err := json.Unmarshal(held.Request, &zeroFields); err != nil {
		t.Fatal(err)
	}
	zeroFields["steps"] = 0
	zero.Request, err = json.Marshal(zeroFields)
	if err != nil {
		t.Fatal(err)
	}
	zeroKey, err := artifact.JSONID(artifact.KindFile, zero.Request)
	if err != nil {
		t.Fatal(err)
	}
	manifest.Draws = map[string]renderedMedia{zeroKey.DigestHex(): zero}
	if err := jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyRendered(t.Context(), store, directory, model, recipeID, zeroKey.DigestHex()); err != nil {
		t.Fatalf("omitted numeric zero changed request semantics: %v", err)
	}
	manifest.Draws = nil
	if err := jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyRendered(t.Context(), store, directory, model, recipeID, name); err == nil {
		t.Error("legacy provenance was qualified")
	}
	manifest.Draws = map[string]renderedMedia{name: held}
	if err := jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyRendered(t.Context(), store, directory, testutil.ArtifactID(t, artifact.KindModel, "another-model"), recipeID, name); err == nil {
		t.Error("a picture another model drew was judged as this one's")
	}
	if _, err := verifyRendered(t.Context(), store, directory, model, testutil.ArtifactID(t, artifact.KindRecipe, "another-recipe"), name); err == nil {
		t.Error("a picture another recipe drew was judged as this one's")
	}
	if _, err := verifyRendered(t.Context(), store, directory, model, recipeID, "unlisted.png"); err == nil {
		t.Error("a picture the manifest does not list was judged")
	}
	if err := os.WriteFile(filepath.Join(directory, held.File), []byte("another picture"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyRendered(t.Context(), store, directory, model, recipeID, name); err == nil {
		t.Error("bytes that are not the digest the file is named for were judged")
	}
	if _, err := verifyRendered(t.Context(), store, t.TempDir(), model, recipeID, name); err == nil {
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
	want := renderManifest{Model: model, Recipe: recipeID, Task: recipe.TaskImageGen, CodeCommit: "new", Draws: map[string]renderedMedia{}}
	fresh, err := alreadyDrawn(t.TempDir(), want)
	if err != nil || len(fresh.Draws) != 0 || fresh.CodeCommit != "new" {
		t.Fatalf("an empty directory gave %+v, %v; want nothing to reuse and this render's commit", fresh, err)
	}
	directory, name, store := renderedDirectory(t, model, recipeID)
	if _, err := verifyRendered(t.Context(), store, directory, model, recipeID, name); err != nil {
		t.Fatal(err)
	}
	held, err := alreadyDrawn(directory, want)
	if err != nil || len(held.Draws) != 1 || held.CodeCommit != strings.Repeat("a", 40) {
		t.Fatalf("a directory this recipe drew gave %+v, %v; want its picture and the commit that drew it", held, err)
	}
	other := want
	other.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "another-recipe")
	if _, err := alreadyDrawn(directory, other); err == nil {
		t.Error("media another recipe drew were reused")
	}
	resized := want
	resized.Settings = map[string]float64{"width": 416}
	if _, err := alreadyDrawn(directory, resized); err == nil {
		t.Error("media drawn with other settings were reused")
	}
	for _, item := range []struct {
		label    string
		prompt   renderPrompt
		settings map[string]float64
		index    int
		same     bool
	}{
		{"same", renderPrompt{ID: "coco_1", Prompt: "a cat"}, nil, 0, true},
		{"case renamed", renderPrompt{ID: "other", Prompt: "a cat"}, nil, 0, true},
		{"prompt", renderPrompt{ID: "coco_1", Prompt: "a dog"}, nil, 0, false},
		{"source", renderPrompt{Prompt: "a cat", Source: testutil.ArtifactID(t, artifact.KindFile, "source")}, nil, 0, false},
		{"seed index", renderPrompt{Prompt: "a cat"}, nil, 1, false},
		{"seed setting", renderPrompt{Prompt: "a cat"}, map[string]float64{"seed": 2}, 0, false},
		{"size", renderPrompt{Prompt: "a cat"}, map[string]float64{"width": 416}, 0, false},
	} {
		_, request, err := renderRequest(map[string]any{"seed": float64(1)}, item.settings, item.prompt, item.index)
		if err != nil {
			t.Fatal(err)
		}
		_, hit := held.Draws[request]
		if hit != item.same {
			t.Errorf("%s reuse=%v want %v", item.label, hit, item.same)
		}
	}
}

// TestAJudgedRuntimeAnswersOnlyAboutAVerifiedPicture holds the judge to its
// contract: it names itself, it asks nothing of a prompt that names no picture,
// and it looks at nothing the manifest does not bind to the evaluated recipe.
func TestAJudgedRuntimeAnswersOnlyAboutAVerifiedPicture(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "drawing-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "drawing-recipe")
	directory, name, store := renderedDirectory(t, model, recipeID)
	judge := judgedRuntime{projectionRuntime: projectionRuntime{mediaFiles: &mediaFiles{root: directory}, identity: evaluation.MediaRuntime{Model: model, Recipe: recipeID}}, model: model, recipe: recipeID, store: store}
	want, _ := artifact.JSONID(artifact.KindProfile, judge.identity)
	if judge.Judge() != want.String() {
		t.Fatalf("the judge names itself %q", judge.Judge())
	}
	input, err := judge.ResolveMedia(t.Context(), evaluation.MediaImage, name)
	if err != nil {
		t.Fatal(err)
	}
	if !input.Producer.Valid() || !input.Request.Valid() {
		t.Fatal("producing request was lost")
	}
	other := judge
	other.identity.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "changed-judge")
	if other.Judge() == judge.Judge() {
		t.Fatal("changed admitted judge recipe kept its identity")
	}
	reference, err := input.Reference()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := judge.mediaFiles.read(reference); err != nil {
		t.Fatal(err)
	}
	bound := judge.files[reference]
	fixed := &mediaFiles{}
	original, err := fixed.bind(bound.path, evaluation.MediaInput{Runtime: judge.identity, Decode: evaluation.MediaDecode{Format: "image"}})
	if err != nil {
		t.Fatal(err)
	}
	copyPath := filepath.Join(t.TempDir(), "relocated.png")
	raw, err := os.ReadFile(bound.path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(copyPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	relocated, err := fixed.bind(copyPath, evaluation.MediaInput{Runtime: judge.identity, Decode: evaluation.MediaDecode{Format: "image"}})
	if err != nil || relocated != original {
		t.Fatalf("identical content relocation changed binding: %v", err)
	}
	if err := os.WriteFile(bound.path, []byte("changed after binding"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := judge.mediaFiles.read(reference); err == nil {
		t.Fatal("changed bytes reached the judge")
	}
	changed, err := fixed.bind(bound.path, evaluation.MediaInput{Runtime: judge.identity, Decode: evaluation.MediaDecode{Format: "image"}})
	if err != nil || changed.Content == original.Content {
		t.Fatalf("changed fixed bytes kept binding: %v", err)
	}
	if _, _, err := judge.Generate(t.Context(), "Which?", inference.GenerateOptions{}); err == nil {
		t.Error("a prompt that names no picture was answered")
	}
	foreign := judge
	foreign.recipe = testutil.ArtifactID(t, artifact.KindRecipe, "another-recipe")
	if _, err := foreign.ResolveMedia(t.Context(), evaluation.MediaImage, name); err == nil {
		t.Error("a picture another recipe drew was answered about")
	}
}
