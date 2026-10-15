package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/atomicfile"
	"overgo/internal/capabilityruntime"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
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

// renderManifest binds full requests to their producing runs and content-addressed files.
type renderManifest struct {
	Model      artifact.ID              `json:"model"`
	Recipe     artifact.ID              `json:"recipe"`
	Task       recipe.Task              `json:"task"`
	CodeCommit string                   `json:"code_commit"`
	Settings   map[string]float64       `json:"settings,omitempty"`
	Draws      map[string]renderedMedia `json:"draws"`
}

// renderedMedia retains the request and original workflow that produced one payload.
// Several requests may share one physical file when their output bytes agree.
type renderedMedia struct {
	File    string          `json:"file"`
	Request json.RawMessage `json:"request"`
	Input   artifact.ID     `json:"input"`
	Output  artifact.ID     `json:"output"`
	Run     artifact.ID     `json:"run,omitzero"`
}

// alreadyDrawn admits a directory only for its declared model, recipe, task and settings.
// Legacy manifests lack request and producer authority and cannot authorize reuse.
func alreadyDrawn(directory string, want renderManifest) (renderManifest, error) {
	data, err := os.ReadFile(filepath.Join(directory, renderManifestFile))
	if errors.Is(err, os.ErrNotExist) {
		return want, nil
	}
	if err != nil {
		return renderManifest{}, err
	}
	var held renderManifest
	if err := json.Unmarshal(data, &held); err != nil {
		return renderManifest{}, fmt.Errorf("evaluate: read the render manifest of %s: %w", directory, err)
	}
	if held.Model != want.Model || held.Recipe != want.Recipe || held.Task != want.Task || !maps.Equal(held.Settings, want.Settings) {
		return renderManifest{}, fmt.Errorf("evaluate: %s holds media drawn by %s under %s with settings %v, not by %s under %s with %v: draw into another directory",
			directory, held.Model, held.Recipe, held.Settings, want.Model, want.Recipe, want.Settings)
	}
	if held.Draws == nil {
		return renderManifest{}, errors.New("evaluate: legacy render provenance is unknown; draw into another directory")
	}
	return held, nil
}

// renderPrompt is one caption to draw and the identity the suite refers to it by.
type renderPrompt struct {
	ID     string
	Prompt string
	// Source is the clip an editor is given to change; a generator draws from the prompt alone.
	Source artifact.ID
}

// renderRequest binds every consumed rendering field before a reuse lookup.
func renderRequest(template map[string]any, settings map[string]float64, item renderPrompt, index int) ([]byte, string, error) {
	request := maps.Clone(template)
	for name, value := range settings {
		request[name] = value
	}
	request["prompt"] = item.Prompt
	if item.Source.Valid() {
		request["source_artifact"] = item.Source
	}
	if seed, seeded := request["seed"].(float64); seeded {
		request["seed"] = seed + float64(index)
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, "", err
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, encoded)
	return encoded, id.DigestHex(), err
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
func drawn(output any) (artifact.Content, string, error) {
	content, err := mediacapability.OutputContent(capabilityruntime.Unwrap(output))
	if err != nil {
		return artifact.Content{}, "", err
	}
	switch content.Descriptor.MediaType {
	case media.PNGMediaType:
		return content, ".png", nil
	case media.GIFMediaType:
		return content, clipExtensionGIF, nil
	default:
		return artifact.Content{}, "", fmt.Errorf("evaluate: output %s is not a PNG or GIF", content.Descriptor.MediaType)
	}
}

// renderMedia draws a picture or clip from each prompt with the model at path under its
// active recipe for task, on the request that recipe's verification ran with the prompt
// and settings put in and the seed advanced for each, writes each to directory named by
// the digest of its bytes, and writes the manifest that binds them to the recipe. It
// returns the request key of each prompt's media. A model whose verified request takes no
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
	manifest, err := alreadyDrawn(directory, renderManifest{
		Model: modelID, Recipe: selected.Activation.Definition.ID, Task: task, CodeCommit: commit, Settings: settings, Draws: map[string]renderedMedia{},
	})
	if err != nil {
		return nil, err
	}
	// The manifest is written after every draw, so a render that is cut short resumes.
	save := func() error {
		return jsonfile.Write(filepath.Join(directory, renderManifestFile), manifest, renderFileMode)
	}
	names := map[string]string{}
	for index, item := range prompts {
		encoded, key, err := renderRequest(template, settings, item, index)
		if err != nil {
			return nil, err
		}
		if _, exists := manifest.Draws[key]; exists {
			if _, err := verifyRendered(ctx, store, directory, manifest.Model, manifest.Recipe, key); err != nil {
				return nil, err
			}
			names[item.ID] = key
			continue
		}
		if manifest.CodeCommit != commit {
			return nil, fmt.Errorf("evaluate: media were drawn at %.12s, this build is %.12s; finish with the producing build or use another directory", manifest.CodeCommit, commit)
		}
		output, err := capability.Execute(capabilityruntime.FreshRun(ctx), store, path, selected, string(encoded))
		if err != nil {
			return nil, fmt.Errorf("evaluate: draw %s: %w", item.ID, err)
		}
		content, extension, err := drawn(output)
		if err != nil || len(content.Data) == 0 {
			return nil, errors.Join(err, fmt.Errorf("evaluate: %s drew nothing for %s", path, item.ID))
		}
		measured, known := output.(capabilityruntime.Measured)
		if !known || measured.Run.Kind() != artifact.KindRun || !measured.Input.Descriptor.ID.Valid() {
			return nil, errors.New("evaluate: rendered output has unknown producing run or request")
		}
		name := content.Descriptor.ID.DigestHex() + extension
		if err := atomicfile.Write(filepath.Join(directory, name), content.Data, renderFileMode); err != nil {
			return nil, err
		}
		manifest.Draws[key] = renderedMedia{File: name, Request: encoded, Input: measured.Input.Descriptor.ID, Output: content.Descriptor.ID, Run: measured.Run}
		names[item.ID] = key
		if err := save(); err != nil {
			return nil, err
		}
		fmt.Printf("  drew %d/%d: %s\n", index+1, len(prompts), item.ID)
	}
	return names, save()
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

// writeDrawnText draws an infographic from each of -cases briefs and writes the suite
// that asks the judge which text each picture holds.
func writeDrawnText(ctx context.Context, model, repository, judge, directory, out, settings string, cases int, seed uint64) error {
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return err
	}
	changes, err := parseSettings(settings)
	if err != nil {
		return err
	}
	items, err := evaluation.PickDrawnText(cases, seed)
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
	suite, err := evaluation.BuildDrawnText(items, images, filepath.Base(judge))
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "drawn-text")
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

const (
	// editSourceFrames frames at editSourceFPS a second are what a clip is cut to for an
	// editor: six latent frames of the video stack, which the editor's chunks of three admit.
	// The size is the editor's own, the one its policy says it was trained at: cut at a
	// quarter of it, the installed editor kept a clip's content and made the change asked
	// in 9 clips of 64.
	editSourceFrames = 21
	editSourceFPS    = 16
	// colorFull is a channel's full value as image colors report it.
	colorFull = 0xffff
)

// publishEditSources cuts each item's clip to the frames an editor takes, at the size the
// editor at model works at, and publishes them to the store as frames, every pixel as it
// was cut, where the editor's request names them. A clip published as the GIF a video
// model returns reaches the editor in 256 colours a frame and smoothed. It returns each
// clip's artifact by its identity.
func publishEditSources(ctx context.Context, repository, clips, model string, items []evaluation.VideoEditItem) (map[string]artifact.ID, error) {
	width, height, err := latentvideo.EditSourceSize(model)
	if err != nil {
		return nil, err
	}
	store, err := overgodb.OpenContext(ctx, repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	sources := map[string]artifact.ID{}
	plane := width * height
	for _, item := range items {
		clip := filepath.Join(clips, item.ID+clipFileExtension)
		frames, err := media.DecodeVideoFileAt(ctx, clip, os.Getenv("OVERGO_FFMPEG"), editSourceFPS, editSourceFrames, width, height)
		if err != nil || len(frames) != editSourceFrames {
			return nil, fmt.Errorf("evaluate: %s gave %d of the %d frames an edit takes: %v", clip, len(frames), editSourceFrames, err)
		}
		source := latentvideo.SourceVideo{
			Pixels: make([]float32, media.RGBChannels*editSourceFrames*plane), Channels: media.RGBChannels,
			Frames: editSourceFrames, Height: height, Width: width,
		}
		for index, frame := range frames {
			origin := frame.Bounds().Min
			for at := range plane {
				red, green, blue, _ := frame.At(origin.X+at%width, origin.Y+at/width).RGBA()
				for channel, value := range []uint32{red, green, blue} {
					source.Pixels[(channel*editSourceFrames+index)*plane+at] = float32(value) / colorFull
				}
			}
		}
		content, err := latentvideo.SourceFramesContent(source)
		if err != nil {
			return nil, err
		}
		if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: "evaluate/edit-source/" + content.Descriptor.ID.String(), Contents: []artifact.Content{content}}); err != nil {
			return nil, err
		}
		sources[item.ID] = content.Descriptor.ID
	}
	return sources, nil
}

// writeVideoEdits picks captioned clips of a directory, has the editor at model change each
// by its instruction, writes the edited clips to directory and the suite that asks the
// judge about them to out. The instruction changes the whole clip, from a fixed list, or,
// with oneThing, the colour of one worn thing the clip's caption names.
func writeVideoEdits(ctx context.Context, clips, captions, model, repository, judge, directory, out string, oneThing bool, cases int, seed uint64) error {
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return err
	}
	if captions == "" {
		captions = filepath.Join(clips, captionFile)
	}
	pick, build, measure := evaluation.PickVideoEdits, evaluation.BuildGeneratedVideoEdit, "generated-video edit"
	if oneThing {
		pick, build, measure = evaluation.PickLocalVideoEdits, evaluation.BuildGeneratedVideoLocalEdit, "generated-video local edit"
	}
	items, err := pick(clips, captions, cases, seed)
	if err != nil {
		return err
	}
	sources, err := publishEditSources(ctx, repository, clips, model, items)
	if err != nil {
		return err
	}
	prompts := make([]renderPrompt, len(items))
	for index, item := range items {
		prompts[index] = renderPrompt{ID: item.ID, Prompt: item.Instruction, Source: sources[item.ID]}
	}
	edited, err := renderMedia(ctx, repository, model, recipe.TaskVideoGen, directory, prompts, nil, commit)
	if err != nil {
		return err
	}
	suite, err := build(items, edited, filepath.Base(judge))
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), measure)
}

// verifyRendered refuses a picture or clip the manifest of its directory does not bind
// to the model and recipe being evaluated: a name the manifest does not list, a
// manifest of another model or recipe, or bytes that are not the digest the file
// is named for.
func verifyRendered(ctx context.Context, store artifact.Reader, directory string, model, recipeID artifact.ID, key string) (renderedMedia, error) {
	var manifest renderManifest
	if err := jsonfile.Decode(filepath.Join(directory, renderManifestFile), &manifest); err != nil {
		return renderedMedia{}, err
	}
	if manifest.Model != model || manifest.Recipe != recipeID {
		return renderedMedia{}, errors.New("evaluate: render model or recipe differs")
	}
	held, found := manifest.Draws[key]
	request, err := artifact.JSONID(artifact.KindFile, held.Request)
	if !found || err != nil || request.DigestHex() != key || held.Run.Kind() != artifact.KindRun || held.Output.Kind() != artifact.KindOutput || !held.Input.Valid() {
		return renderedMedia{}, errors.New("evaluate: render request or producer provenance is unknown")
	}
	path, err := mediaPath(directory, held.File)
	if err != nil {
		return renderedMedia{}, err
	}
	content, err := artifact.IdentifyFile(artifact.KindOutput, path)
	if err != nil || content != held.Output {
		return renderedMedia{}, errors.Join(err, errors.New("evaluate: rendered payload differs"))
	}
	produced, err := runrecord.RequireRun(ctx, store, held.Run)
	if err != nil || produced.Outcome != runrecord.OutcomeSucceeded || produced.Recipe != recipeID || produced.CodeCommit != manifest.CodeCommit || !slices.Contains(produced.Inputs, held.Input) || !slices.Contains(produced.Outputs, held.Output) {
		return renderedMedia{}, errors.Join(err, errors.New("evaluate: producing run does not bind this request and output"))
	}
	input, found, err := artifact.ReadContent(ctx, store, held.Input)
	if err != nil || !found {
		return renderedMedia{}, errors.Join(err, errors.New("evaluate: producing request is absent"))
	}
	retained, known := capabilityruntime.RetainedRequest(input)
	if !known {
		return renderedMedia{}, errors.New("evaluate: producing request is not retained JSON")
	}
	documents := []map[string]any{{}, {}}
	for index, raw := range [][]byte{held.Request, []byte(retained)} {
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		if err := decoder.Decode(&documents[index]); err != nil {
			return renderedMedia{}, err
		}
		// Numeric zero is the omitted scalar value of a typed capability request.
		for field, value := range documents[index] {
			if value == json.Number("0") || value == json.Number("-0") {
				delete(documents[index], field)
			}
		}
	}
	if !reflect.DeepEqual(documents[0], documents[1]) {
		return renderedMedia{}, errors.New("evaluate: complete producing request differs")
	}
	return held, nil
}
