package main

import (
	"bytes"
	"cmp"
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
	"overgo/internal/envsetting"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
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
	// gifFormat is the format a clip's request names by naming none, and a render manifest written before formats were
	// recorded drew its clips in.
	gifFormat = "gif"
)

// drawnExtensions name each format a generation returns by the extension its file is written under: a picture as a
// PNG, a clip as the GIF or MP4 it was written as, which the judge reads natively or through FFmpeg.
var drawnExtensions = map[string]string{media.PNGMediaType: ".png", media.GIFMediaType: ".gif", media.MP4MediaType: ".mp4"}

// renderManifest binds full requests to their producing runs and content-addressed files.
type renderManifest struct {
	Model      artifact.ID        `json:"model"`
	Recipe     artifact.ID        `json:"recipe"`
	Task       recipe.Task        `json:"task"`
	CodeCommit string             `json:"code_commit"`
	Settings   map[string]float64 `json:"settings,omitempty"`
	// Preset is the preset of its model the media were drawn with, none for its reference, the authors' recipe
	// (judge-calibration/user-size): a preset's arm and its reference's are drawn into directories of their own.
	Preset string `json:"preset,omitempty"`
	// Format is the format the clips were asked in (clipFormat); a picture render holds none.
	Format string                   `json:"format,omitempty"`
	Draws  map[string]renderedMedia `json:"draws"`
}

// clipFormat is the format a render's clips were written in: the one its manifest names, or a GIF for clips drawn
// before a manifest named one, which is what their requests asked for by naming none. Two renders compared one beside
// the other (a pair screen, a model beside its base) agree in it, or the comparison measures the format with the model.
func (manifest renderManifest) clipFormat() string {
	if manifest.Format == "" && manifest.Task == recipe.TaskVideoGen {
		return gifFormat
	}
	return manifest.Format
}

// renderedMedia retains the request and original workflow that produced one payload.
// Several requests may share one physical file when their output bytes agree.
type renderedMedia struct {
	File    string          `json:"file"`
	Request json.RawMessage `json:"request"`
	Input   artifact.ID     `json:"input"`
	Output  artifact.ID     `json:"output"`
	Run     artifact.ID     `json:"run,omitzero"`
	// Encoding is the FFmpeg and the arguments that wrote an MP4 (media.EncodingOf); a GIF or a picture holds none.
	Encoding media.Encoding `json:"encoding,omitzero"`
	// Build is the build that drew it, where it is not the manifest's (sameAtThisBuild); none is the manifest's.
	Build string `json:"build,omitempty"`
}

// alreadyDrawn admits a directory only for its declared model, recipe, task, settings and clip format.
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
	if held.Preset != want.Preset {
		return renderManifest{}, fmt.Errorf("evaluate: %s holds media drawn with preset %q, and this render draws with %q: draw into another directory", directory, held.Preset, want.Preset)
	}
	if held.clipFormat() != want.clipFormat() {
		return renderManifest{}, fmt.Errorf("evaluate: %s holds clips written as %s, and this render writes %s: draw into another directory", directory, held.clipFormat(), want.clipFormat())
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
	extension, written := drawnExtensions[content.Descriptor.MediaType]
	if !written {
		return artifact.Content{}, "", fmt.Errorf("evaluate: output %s is not a PNG, GIF or MP4", content.Descriptor.MediaType)
	}
	return content, extension, nil
}

// withDeclaredDefaults gives the request a render draws from each field its page's controls declare a default for
// and it leaves out, as the server's intake gives a person's request (mediacapability.Control.Default): a clip asked
// for in no format is drawn as an MP4, and the request records it.
func withDeclaredDefaults(selected modelrecipe.CapabilityEvidenceSelection, directory string, template map[string]any) {
	stages := selected.Program.Stages()
	if len(stages) == 0 {
		return
	}
	controls, _ := mediacapability.Controls(stages[0].Module.ID, directory)
	for _, control := range controls {
		// A control that applies with another field is given nothing where the request does not ask for that field
		// (mediacapability.Control.With): a clip drawn at its own size is not given an upscale's level.
		if asked := template[control.With]; control.With != "" && (asked == nil || reflect.ValueOf(asked).IsZero()) {
			continue
		}
		if _, stated := template[control.Name]; control.Default != "" && (!stated || template[control.Name] == nil) {
			template[control.Name] = control.Default
		}
	}
}

// renderMedia draws a picture or clip from each prompt with the model at path under its
// active recipe for task, on the request that recipe's verification ran with the prompt
// and settings put in and the seed advanced for each, writes each to directory named by
// the digest of its bytes, and writes the manifest that binds them to the recipe. It
// returns the request key of each prompt's media. A model whose verified request takes no
// prompt draws nothing from a caption and is refused.
func renderMedia(ctx context.Context, repository, path string, task recipe.Task, directory string, prompts []renderPrompt, settings map[string]float64, preset, commit string) (names map[string]string, err error) {
	store, err := overgodb.OpenContext(ctx, repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	// The model stays loaded from one draw to the next, its session given back when the suite is drawn: a
	// request-scoped recipe had reloaded MOSS for each picture, about 10 s of each 70 s one (guidance-ops/held-renders).
	ctx = capabilityruntime.HoldSessions(ctx)
	defer func() {
		_, releaseErr := capabilityruntime.ReleaseIdleSessions(context.WithoutCancel(ctx))
		err = errors.Join(err, releaseErr)
	}()
	capability := mediacapability.Catalog[task]
	source, err := capability.Resolve(ctx, path)
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
	withDeclaredDefaults(selected, path, template)
	if template, err = personSized(selected, path, template, settings); err != nil {
		return nil, err
	}
	if preset != "" {
		template["preset"] = preset
	}
	if err := os.MkdirAll(directory, renderDirectoryMode); err != nil {
		return nil, err
	}
	want := renderManifest{
		Model: modelID, Recipe: selected.Activation.Definition.ID, Task: task, CodeCommit: commit, Settings: settings, Preset: preset, Draws: map[string]renderedMedia{},
	}
	if format, named := template["format"].(string); named && task == recipe.TaskVideoGen {
		want.Format = format
	}
	manifest, err := alreadyDrawn(directory, want)
	if err != nil {
		return nil, err
	}
	set := renderSet{store: store, directory: directory, manifest: manifest, commit: commit,
		request: func(item renderPrompt, index int) ([]byte, string, error) {
			return renderRequest(template, settings, item, index)
		},
		draw: func(encoded []byte) (drawnMedia, error) {
			output, err := capability.Execute(capabilityruntime.FreshRun(ctx), store, path, selected, string(encoded))
			if err != nil {
				return drawnMedia{}, err
			}
			content, extension, err := drawn(output)
			if err != nil || len(content.Data) == 0 {
				return drawnMedia{}, errors.Join(err, fmt.Errorf("evaluate: %s drew nothing", path))
			}
			measured, known := output.(capabilityruntime.Measured)
			if !known || measured.Run.Kind() != artifact.KindRun || !measured.Input.Descriptor.ID.Valid() {
				return drawnMedia{}, errors.New("evaluate: rendered output has unknown producing run or request")
			}
			return drawnMedia{data: content.Data, extension: extension, entry: renderedMedia{
				Request: encoded, Input: measured.Input.Descriptor.ID, Output: content.Descriptor.ID, Run: measured.Run,
				Encoding: media.EncodingOf(capabilityruntime.Unwrap(output)),
			}}, nil
		}}
	return set.drawEach(ctx, prompts)
}

// drawnMedia is one draw's bytes, the extension its file is written under and the entry the manifest binds it by,
// its file and build not yet named.
type drawnMedia struct {
	data      []byte
	extension string
	entry     renderedMedia
}

// renderSet is a directory's render as it is drawn: the manifest that binds what it holds, the build that draws now,
// the request each prompt makes and the draw of one.
type renderSet struct {
	store     artifact.Reader
	directory string
	manifest  renderManifest
	commit    string
	request   func(item renderPrompt, index int) (encoded []byte, key string, err error)
	draw      func(encoded []byte) (drawnMedia, error)
}

// drawEach takes each prompt's media the manifest holds and draws the rest, each draw naming its build. Where the
// last item taken was drawn at another build, it is drawn again before the next is drawn, and the set continues only
// where the bytes are the same (sameAtThisBuild).
func (s *renderSet) drawEach(ctx context.Context, prompts []renderPrompt) (map[string]string, error) {
	// The manifest is written after every draw, so a render that is cut short resumes.
	save := func() error {
		return jsonfile.Write(filepath.Join(s.directory, renderManifestFile), s.manifest, renderFileMode)
	}
	names := map[string]string{}
	var last renderPrompt
	for index, item := range prompts {
		encoded, key, err := s.request(item, index)
		if err != nil {
			return nil, err
		}
		if _, exists := s.manifest.Draws[key]; exists {
			if _, err := verifyRendered(ctx, s.store, s.directory, s.manifest.Model, s.manifest.Recipe, key); err != nil {
				return nil, err
			}
			names[item.ID], last = key, item
			continue
		}
		if last.ID != "" {
			if err := s.sameAtThisBuild(last.ID, names[last.ID]); err != nil {
				return nil, err
			}
			last = renderPrompt{}
		}
		drawn, err := s.draw(encoded)
		if err != nil {
			return nil, fmt.Errorf("evaluate: draw %s: %w", item.ID, err)
		}
		name := drawn.entry.Output.DigestHex() + drawn.extension
		if err := atomicfile.Write(filepath.Join(s.directory, name), drawn.data, renderFileMode); err != nil {
			return nil, err
		}
		drawn.entry.File, drawn.entry.Build = name, s.commit
		s.manifest.Draws[key] = drawn.entry
		names[item.ID] = key
		if err := save(); err != nil {
			return nil, err
		}
		fmt.Printf("  drew %d/%d: %s\n", index+1, len(prompts), item.ID)
		// A waiter for the card says how far the render has come, by this count (card-handover/counts).
		if err := processcontrol.TurnCount(index+1, len(prompts)); err != nil {
			return nil, err
		}
		if gave, err := giveTheCardToAWaiter(ctx, capabilityruntime.ReleaseIdleSessions); err != nil {
			return nil, err
		} else if gave {
			fmt.Printf("  gave the card to a waiting process after %d/%d\n", index+1, len(prompts))
		}
	}
	return names, save()
}

// sameAtThisBuild draws the item named again where another build drew it, and refuses the render where this build
// draws other bytes, naming the item and both builds: a set drawn at one build continues at another only where the
// new build draws the same, so a fix to code the set does not run no longer strands it (render-resume/same-bytes),
// where MOSS's APG set stood at 40 of 80 and Wan's reference at 35 of 100 on 2026-11-12, each refused by the build
// alone.
func (s *renderSet) sameAtThisBuild(item, key string) error {
	held := s.manifest.Draws[key]
	build := cmp.Or(held.Build, s.manifest.CodeCommit)
	if build == s.commit {
		return nil
	}
	again, err := s.draw(held.Request)
	if err != nil {
		return fmt.Errorf("evaluate: draw %s again at %.12s: %w", item, s.commit, err)
	}
	if again.entry.Output != held.Output {
		return fmt.Errorf("evaluate: %s drawn at %.12s is drawn otherwise at this build %.12s; finish with the producing build or use another directory", item, build, s.commit)
	}
	fmt.Printf("  drew %s again at %.12s the same as at %.12s: the set continues\n", item, s.commit, build)
	return nil
}

// giveTheCardToAWaiter closes the sessions release closes where another process waits for the card's turn this one
// holds (processcontrol.TurnWaited), so the turn passes to it between pictures, a held session's contexts holding it,
// and the next picture loads them again in its turn (card-turn/readings-turn); with no waiter they stay loaded. A
// release that leaves the turn held refuses the render, naming it: a Wan render said it gave the card up after each of
// 31 videos while its text encoder, kept outside its sessions, held the turn, and a landing's test waited 3.5 hours
// (card-handover/encoder-release).
func giveTheCardToAWaiter(ctx context.Context, release func(context.Context) (int, error)) (bool, error) {
	waited, err := processcontrol.TurnWaited()
	if err != nil || !waited {
		return false, err
	}
	closed, err := release(ctx)
	if err != nil {
		return false, err
	}
	// The waiter cannot take a turn this process holds, so one still waiting waits for this process.
	if held, err := processcontrol.TurnWaited(); err != nil || !held {
		return err == nil, err
	}
	return false, fmt.Errorf("evaluate: a process waits for the card, and closing %d idle sessions left this render holding its turn: a context is kept outside them", closed)
}

// composedShare is how many captions a picture suite picks for each it composes of 3 to 10 atoms beside them
// (evaluation.ComposeTIFA, evaluation.ComposeDrawnText): the picked ones keep the measure the suite had, and the
// composed ones keep it off a ceiling a served model reached.
const composedShare = 4

// writeTIFAImages draws the pictures of -cases TIFA captions and writes the suite
// that asks the judge about them.
func writeTIFAImages(ctx context.Context, questions, model, repository, judge, directory, out, settings, preset string, cases int, seed uint64) error {
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
	composed, err := evaluation.ComposeTIFA(max(cases/composedShare, 1), seed)
	if err != nil {
		return err
	}
	items = append(items, composed...)
	prompts := make([]renderPrompt, len(items))
	for index, item := range items {
		prompts[index] = renderPrompt{ID: item.ID, Prompt: item.Caption}
	}
	images, err := renderMedia(ctx, repository, model, recipe.TaskImageGen, directory, prompts, changes, preset, commit)
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
func writeDrawnText(ctx context.Context, model, repository, judge, directory, out, settings, preset string, cases int, seed uint64) error {
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
	composed, err := evaluation.ComposeDrawnText(max(cases/composedShare, 1), seed)
	if err != nil {
		return err
	}
	items = append(items, composed...)
	prompts := make([]renderPrompt, len(items))
	for index, item := range items {
		prompts[index] = renderPrompt{ID: item.ID, Prompt: item.Caption}
	}
	images, err := renderMedia(ctx, repository, model, recipe.TaskImageGen, directory, prompts, changes, preset, commit)
	if err != nil {
		return err
	}
	suite, err := evaluation.BuildDrawnText(items, images, filepath.Base(judge))
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "drawn-text")
}

// writeVidGenClips draws the clips of -cases VidGen captions and of the captions composed
// beside them, and writes the suite that asks the judge which of four captions each picked
// one shows and what each composed one holds.
func writeVidGenClips(ctx context.Context, captions, model, repository, judge, directory, out, settings, preset string, cases int, seed uint64) error {
	commit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return err
	}
	changes, err := parseSettings(settings)
	if err != nil {
		return err
	}
	items, composed, prompts, err := vidGenCaptions(captions, cases, seed)
	if err != nil {
		return err
	}
	clips, err := renderMedia(ctx, repository, model, recipe.TaskVideoGen, directory, prompts, changes, preset, commit)
	if err != nil {
		return err
	}
	suite, err := evaluation.BuildGeneratedVideoMatching(items, composed, clips, filepath.Base(judge))
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "generated-video caption-matching")
}

// vidGenCaptions picks -cases VidGen captions by seed and composes one caption for each
// composedShare of them (evaluation.ComposeTIFA), as a picture suite does, and returns the
// prompts a video model draws: the picked captions, then the composed ones.
func vidGenCaptions(captions string, cases int, seed uint64) ([]evaluation.VideoCaptionItem, []evaluation.TIFAItem, []renderPrompt, error) {
	items, err := evaluation.PickVideoCaptions(captions, cases, seed)
	if err != nil {
		return nil, nil, nil, err
	}
	composed, err := evaluation.ComposeTIFA(max(cases/composedShare, 1), seed)
	if err != nil {
		return nil, nil, nil, err
	}
	var prompts []renderPrompt
	for _, item := range items {
		prompts = append(prompts, renderPrompt{ID: item.ID, Prompt: item.Caption})
	}
	for _, item := range composed {
		prompts = append(prompts, renderPrompt{ID: item.ID, Prompt: item.Caption})
	}
	return items, composed, prompts, nil
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
		frames, err := media.DecodeVideoFileAt(ctx, clip, envsetting.Result("OVERGO_FFMPEG"), editSourceFPS, editSourceFrames, width, height)
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
	edited, err := renderMedia(ctx, repository, model, recipe.TaskVideoGen, directory, prompts, nil, "", commit)
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
	// A capability run records no commit; one that does names the build that drew it.
	otherBuild := produced.CodeCommit != "" && produced.CodeCommit != cmp.Or(held.Build, manifest.CodeCommit)
	if err != nil || produced.Outcome != runrecord.OutcomeSucceeded || produced.Recipe != recipeID || otherBuild || !slices.Contains(produced.Inputs, held.Input) || !slices.Contains(produced.Outputs, held.Output) {
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
