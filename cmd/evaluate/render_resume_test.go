package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestARenderContinuesAtABuildThatDrawsItsLastItemTheSame holds a render resumed at another build to the bytes of the
// last item it takes from its directory: a build that draws that item the same draws on, its draw naming the build;
// one that draws it otherwise refuses before drawing anything more, naming the item and both builds; and the build
// that drew it draws on without drawing it again.
func TestARenderContinuesAtABuildThatDrawsItsLastItemTheSame(t *testing.T) {
	t.Parallel()
	drawnAt, next := strings.Repeat("a", 40), strings.Repeat("b", 40)
	prompts := []renderPrompt{{ID: "coco_1", Prompt: "a cat"}, {ID: "coco_2", Prompt: "a dog"}}
	for _, test := range []struct {
		name, commit, cat string
		draws             int
		refused           bool
	}{
		{"same bytes at another build", next, "a picture the model drew", 2, false},
		{"other bytes at another build", next, "a cat drawn otherwise", 1, true},
		{"the build that drew it", drawnAt, "a cat never drawn again", 1, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			model := testutil.ArtifactID(t, artifact.KindModel, "drawing-model")
			recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "drawing-recipe")
			directory, _, store := renderedDirectory(t, model, recipeID)
			manifest := testcheck.Must(alreadyDrawn(directory, renderManifest{Model: model, Recipe: recipeID, Task: recipe.TaskImageGen, CodeCommit: test.commit, Draws: map[string]renderedMedia{}}))(t)
			var drawn []string
			input, run := testutil.ArtifactID(t, artifact.KindFile, "request"), testutil.ArtifactID(t, artifact.KindRun, "run")
			set := renderSet{store: store, directory: directory, manifest: manifest, commit: test.commit,
				request: func(item renderPrompt, index int) ([]byte, string, error) {
					return renderRequest(map[string]any{"seed": float64(1)}, nil, item, index)
				},
				draw: func(encoded []byte) (drawnMedia, error) {
					picture := []byte("a dog the model drew")
					if strings.Contains(string(encoded), `"a cat"`) {
						picture = []byte(test.cat)
					}
					drawn = append(drawn, string(picture))
					output, err := artifact.IdentifyBytes(artifact.KindOutput, picture)
					return drawnMedia{data: picture, extension: ".png", entry: renderedMedia{Request: encoded, Input: input, Output: output, Run: run}}, err
				}}
			names, err := set.drawEach(t.Context(), prompts)
			if len(drawn) != test.draws {
				t.Fatalf("the render drew %q, want %d draws", drawn, test.draws)
			}
			var held renderManifest
			testcheck.NoError(t, jsonfile.Decode(filepath.Join(directory, renderManifestFile), &held))
			if test.refused {
				for _, named := range []string{"coco_1", drawnAt[:12], next[:12]} {
					if err == nil || !strings.Contains(err.Error(), named) {
						t.Fatalf("a build that draws the last item otherwise = %v, want a refusal naming %s", err, named)
					}
				}
				if len(held.Draws) != 1 {
					t.Fatalf("the refused render's manifest holds %d draws, want the one drawn before", len(held.Draws))
				}
				return
			}
			testcheck.NoError(t, err)
			if dog := held.Draws[names["coco_2"]]; dog.Build != test.commit || held.Draws[names["coco_1"]].Build != "" {
				t.Fatalf("the draws name builds %q and %q, want the one drawn before its manifest's and the next %.12s", held.Draws[names["coco_1"]].Build, dog.Build, test.commit)
			}
		})
	}
}
