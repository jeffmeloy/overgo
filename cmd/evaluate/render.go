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
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/evaluation"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const (
	// renderManifestFile names, in a media directory, the manifest that says which model's
	// recipe drew its pictures and clips and from what.
	renderManifestFile = "rendered.json"
	// renderFileMode and renderDirectoryMode are the permissions of what a render writes.
	renderFileMode      = 0o600
	renderDirectoryMode = 0o700
	// settingBits is the precision a -render-set number is read at, a request's JSON number.
	settingBits = 64
	// clipExtensionGIF is the extension of a clip a video model draws, which the judge reads natively.
	clipExtensionGIF = ".gif"
)

// renderManifest binds a directory of pictures or clips to the model and recipe that drew
// them, the commit of the code that ran them, and the caption each was drawn from, by the
// file name, which is the digest of its bytes and its extension.
type renderManifest struct {
	Model      artifact.ID       `json:"model"`
	Recipe     artifact.ID       `json:"recipe"`
	Task       recipe.Task       `json:"task"`
	CodeCommit string            `json:"code_commit"`
	Files      map[string]string `json:"files"`
}

// renderPrompt is one caption to draw and the identity the suite refers to it by.
type renderPrompt struct {
	ID     string
	Prompt string
}

// verifiedTemplate is the request a recipe's accepted verification ran, which the media
// are drawn from with another prompt: its size, steps and sampler are the ones the recipe
// was proved on.
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

// parseSettings reads -render-set's comma-separated name=number settings, which replace
// fields of the request a render draws from: the size a verification smoke ran at is
// too small for a judge to see anything in.
func parseSettings(text string) (map[string]float64, error) {
	settings := map[string]float64{}
	for setting := range strings.SplitSeq(text, ",") {
		if strings.TrimSpace(setting) == "" {
			continue
		}
		name, value, found := strings.Cut(setting, "=")
		number, err := strconv.ParseFloat(strings.TrimSpace(value), settingBits)
		if !found || strings.TrimSpace(name) == "" || err != nil {
			return nil, fmt.Errorf("evaluate: setting %q is not name=number", setting)
		}
		settings[strings.TrimSpace(name)] = number
	}
	return settings, nil
}

// drawn is what a generation capability returned: the bytes to write and the extension
// that names their format.
func drawn(output any) ([]byte, string, error) {
	switch result := capabilityruntime.Unwrap(output).(type) {
	case latentimage.EncodedImage:
		return result.Data, ".png", nil
	case latentvideo.EncodedVideo:
		if result.MediaType != media.GIFMediaType {
			return nil, "", fmt.Errorf("evaluate: a clip of %s is not the GIF the judge reads", result.MediaType)
		}
		return result.Data, clipExtensionGIF, nil
	default:
		return nil, "", fmt.Errorf("evaluate: the model drew %T, not an encoded picture or clip", result)
	}
}

// renderMedia draws a picture or clip from each prompt with the model at path under its
// active recipe for task, on the request that recipe's verification ran with the prompt
// and settings put in and the seed advanced for each, writes each to directory named by
// the digest of its bytes, and writes the manifest that binds them to the recipe. It
// returns the file name of each prompt's media. A model whose verified request takes no
// prompt draws nothing from a caption and is refused.
func renderMedia(ctx context.Context, repository, path string, task recipe.Task, directory string, prompts []renderPrompt, settings map[string]float64, commit string) (map[string]string, error) {
	store, err := overgodb.OpenContext(ctx, repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	capability := mediacapability.Catalog[task]
	source, err := capability.Resolve(path)
	if err != nil {
		return nil, err
	}
	modelID := source.Inventory.Manifest.ID
	selected, err := modelrecipe.ResolveActiveExecution(ctx, store, modelID, task, modelrecipe.SessionWarm)
	if err != nil {
		return nil, err
	}
	template, err := verifiedTemplate(ctx, store, selected.Activation)
	if err != nil {
		return nil, err
	}
	if _, prompted := template["prompt"].(string); !prompted {
		return nil, fmt.Errorf("evaluate: %s draws nothing from a caption: its verified request holds no prompt", path)
	}
	if err := os.MkdirAll(directory, renderDirectoryMode); err != nil {
		return nil, err
	}
	manifest := renderManifest{Model: modelID, Recipe: selected.Activation.Definition.ID, Task: task, CodeCommit: commit, Files: map[string]string{}}
	names := map[string]string{}
	for index, item := range prompts {
		request := make(map[string]any, len(template)+len(settings))
		maps.Copy(request, template)
		for name, value := range settings {
			request[name] = value
		}
		request["prompt"] = item.Prompt
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
		data, extension, err := drawn(output)
		if err != nil || len(data) == 0 {
			return nil, errors.Join(err, fmt.Errorf("evaluate: %s drew nothing for %s", path, item.ID))
		}
		sum := sha256.Sum256(data)
		name := hex.EncodeToString(sum[:]) + extension
		if err := os.WriteFile(filepath.Join(directory, name), data, renderFileMode); err != nil {
			return nil, err
		}
		manifest.Files[name], names[item.ID] = item.ID, name
		fmt.Printf("  drew %d/%d: %s\n", index+1, len(prompts), item.ID)
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, err
	}
	return names, os.WriteFile(filepath.Join(directory, renderManifestFile), data, renderFileMode)
}

// writeTIFAImages draws the pictures of -cases TIFA captions and writes the suite
// that asks the judge about them.
func writeTIFAImages(ctx context.Context, questions, model, repository, judge, directory, out, settings string, cases int, seed uint64) error {
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return err
	}
	changes, err := parseSettings(settings)
	if err != nil {
		return err
	}
	items, err := evaluation.PickTIFA(questions, cases, seed)
	if err != nil {
		return err
	}
	prompts := make([]renderPrompt, len(items))
	for index, item := range items {
		prompts[index] = renderPrompt{ID: item.ID, Prompt: item.Caption}
	}
	images, err := renderMedia(ctx, repository, model, recipe.TaskImageGen, directory, prompts, changes, commit)
	if err != nil {
		return err
	}
	suite, err := evaluation.BuildTIFAQuestions(items, images, filepath.Base(judge))
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "TIFA image-question")
}

// writeVidGenClips draws the clips of -cases VidGen captions and writes the suite that
// asks the judge which of four captions each shows.
func writeVidGenClips(ctx context.Context, captions, model, repository, judge, directory, out, settings string, cases int, seed uint64) error {
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return err
	}
	changes, err := parseSettings(settings)
	if err != nil {
		return err
	}
	items, err := evaluation.PickVideoCaptions(captions, cases, seed)
	if err != nil {
		return err
	}
	prompts := make([]renderPrompt, len(items))
	for index, item := range items {
		prompts[index] = renderPrompt{ID: item.ID, Prompt: item.Caption}
	}
	clips, err := renderMedia(ctx, repository, model, recipe.TaskVideoGen, directory, prompts, changes, commit)
	if err != nil {
		return err
	}
	suite, err := evaluation.BuildGeneratedVideoMatching(items, clips, filepath.Base(judge))
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "generated-video caption-matching")
}

// verifyRendered refuses a picture or clip the manifest of its directory does not bind
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
		return fmt.Errorf("evaluate: the media of %s were drawn by %s under %s, not by %s under %s", directory, manifest.Model, manifest.Recipe, model, recipeID)
	}
	if _, listed := manifest.Files[name]; !listed {
		return fmt.Errorf("evaluate: the render manifest of %s does not list %s", directory, name)
	}
	content, err := os.ReadFile(filepath.Join(directory, name))
	if err != nil {
		return err
	}
	sum := sha256.Sum256(content)
	if digest, _, _ := strings.Cut(name, "."); hex.EncodeToString(sum[:]) != digest {
		return errors.New("evaluate: " + name + " is not the media its digest names")
	}
	return nil
}
