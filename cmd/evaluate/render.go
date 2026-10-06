package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/evaluation"
	"overgo/internal/latentimage"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const (
	// renderManifestFile names, in a media directory, the manifest that says which model's
	// recipe drew its pictures and from what.
	renderManifestFile = "rendered.json"
	// renderFileMode and renderDirectoryMode are the permissions of what a render writes.
	renderFileMode      = 0o600
	renderDirectoryMode = 0o700
)

// renderManifest binds a directory of pictures to the model and recipe that drew
// them, the commit of the code that ran them, and the caption each was drawn
// from, by the digest its file is named for.
type renderManifest struct {
	Model      artifact.ID       `json:"model"`
	Recipe     artifact.ID       `json:"recipe"`
	Task       recipe.Task       `json:"task"`
	CodeCommit string            `json:"code_commit"`
	Files      map[string]string `json:"files"`
}

// verifiedTemplate is the request a recipe's accepted verification ran, which the
// pictures are drawn from with another prompt: its size, steps and sampler are the
// ones the recipe was proved on.
func verifiedTemplate(ctx context.Context, store artifact.Reader, activation modelrecipe.Activation) (map[string]any, error) {
	for _, input := range activation.Verified.Run.Inputs {
		content, found, err := artifact.ReadContent(ctx, store, input)
		if err != nil {
			return nil, err
		}
		if request, retained := capabilityruntime.RetainedRequest(content); found && retained {
			var template map[string]any
			if err := json.Unmarshal([]byte(request), &template); err != nil {
				return nil, fmt.Errorf("evaluate: the verified request of %s is no JSON object: %w", activation.Definition.ID, err)
			}
			return template, nil
		}
	}
	return nil, fmt.Errorf("evaluate: the accepted verification run of %s retained no runnable request", activation.Definition.ID)
}

// renderImages draws a picture from each TIFA caption with the model at path
// under its active image-gen recipe, writes each to directory named by the digest
// of its PNG bytes, and writes the manifest that binds them to the recipe. A model
// whose verified request takes no prompt draws nothing from a caption and is refused.
func renderImages(ctx context.Context, repository, path, directory string, items []evaluation.TIFAItem, commit string) (map[string]string, error) {
	store, err := overgodb.OpenContext(ctx, repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	capability := mediacapability.Catalog[recipe.TaskImageGen]
	source, err := capability.Resolve(path)
	if err != nil {
		return nil, err
	}
	modelID := source.Inventory.Manifest.ID
	selected, err := modelrecipe.ResolveActiveExecution(ctx, store, modelID, recipe.TaskImageGen, modelrecipe.SessionWarm)
	if err != nil {
		return nil, err
	}
	template, err := verifiedTemplate(ctx, store, selected.Activation)
	if err != nil {
		return nil, err
	}
	if _, prompted := template["prompt"].(string); !prompted {
		return nil, fmt.Errorf("evaluate: %s draws no picture from a caption: its verified request holds no prompt", path)
	}
	if err := os.MkdirAll(directory, renderDirectoryMode); err != nil {
		return nil, err
	}
	manifest := renderManifest{Model: modelID, Recipe: selected.Activation.Definition.ID, Task: recipe.TaskImageGen, CodeCommit: commit, Files: map[string]string{}}
	images := map[string]string{}
	for index, item := range items {
		request := make(map[string]any, len(template))
		maps.Copy(request, template)
		request["prompt"] = item.Caption
		if seed, seeded := template["seed"].(float64); seeded {
			request["seed"] = seed + float64(index)
		}
		encoded, err := json.Marshal(request)
		if err != nil {
			return nil, err
		}
		output, err := capability.Execute(capabilityruntime.FreshRun(ctx), store, path, selected, string(encoded))
		if err != nil {
			return nil, fmt.Errorf("evaluate: draw %s: %w", item.ID, err)
		}
		picture, drawn := capabilityruntime.Unwrap(output).(latentimage.EncodedImage)
		if !drawn || len(picture.Data) == 0 {
			return nil, fmt.Errorf("evaluate: %s drew %T for %s, not an encoded image", path, capabilityruntime.Unwrap(output), item.ID)
		}
		sum := sha256.Sum256(picture.Data)
		digest := hex.EncodeToString(sum[:])
		if err := os.WriteFile(filepath.Join(directory, digest+".png"), picture.Data, renderFileMode); err != nil {
			return nil, err
		}
		manifest.Files[digest+".png"], images[item.ID] = item.ID, digest
		fmt.Printf("  drew %d/%d: %s\n", index+1, len(items), item.ID)
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, err
	}
	return images, os.WriteFile(filepath.Join(directory, renderManifestFile), data, renderFileMode)
}

// writeTIFAImages draws the pictures of -cases TIFA captions and writes the suite
// that asks the judge about them.
func writeTIFAImages(ctx context.Context, questions, model, repository, judge, directory, out string, cases int, seed uint64) error {
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return err
	}
	items, err := evaluation.PickTIFA(questions, cases, seed)
	if err != nil {
		return err
	}
	images, err := renderImages(ctx, repository, model, directory, items, commit)
	if err != nil {
		return err
	}
	suite, err := evaluation.BuildTIFAQuestions(items, images, filepath.Base(judge))
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "TIFA image-question")
}

// verifyRendered refuses a picture the manifest of its directory does not bind
// to the model and recipe being evaluated: a name the manifest does not list, a
// manifest of another model or recipe, or bytes that are not the digest the file
// is named for.
func verifyRendered(directory string, model, recipeID artifact.ID, name string) error {
	data, err := os.ReadFile(filepath.Join(directory, renderManifestFile))
	if err != nil {
		return fmt.Errorf("evaluate: %s holds no render manifest: %w", directory, err)
	}
	var manifest renderManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("evaluate: read the render manifest of %s: %w", directory, err)
	}
	if manifest.Model != model || manifest.Recipe != recipeID {
		return fmt.Errorf("evaluate: the pictures of %s were drawn by %s under %s, not by %s under %s", directory, manifest.Model, manifest.Recipe, model, recipeID)
	}
	if _, listed := manifest.Files[name]; !listed {
		return fmt.Errorf("evaluate: the render manifest of %s does not list %s", directory, name)
	}
	picture, err := os.ReadFile(filepath.Join(directory, name))
	if err != nil {
		return err
	}
	sum := sha256.Sum256(picture)
	if digest, _, _ := strings.Cut(name, "."); hex.EncodeToString(sum[:]) != digest {
		return errors.New("evaluate: " + name + " is not the picture its digest names")
	}
	return nil
}
