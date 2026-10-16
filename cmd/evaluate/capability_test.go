package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/seriesforecast"
	"overgo/internal/testutil"
)

// capabilityOpening is one opening of a capability session over a model
// published for a test: what the opener returned, how often it loaded and
// released the runtime it was handed, and whether it wrote to the store.
type capabilityOpening struct {
	session         evaluationSession
	err             error
	loads, releases int
	wrote           bool
}

// capabilityDefiner is the recipe a test gives a published model, over the
// profile published beside it where the recipe depends on one.
type capabilityDefiner func(model, profile artifact.ID) (recipe.Definition, error)

// openPublishedCapability publishes a one-file model and a profile in a store
// of its own, takes the recipe define gives the model through activation when
// define is given, and opens a capability session over it for task with a
// runtime that loads nothing. The release of that runtime reports released.
func openPublishedCapability(t *testing.T, value manifest, task recipe.Task, define capabilityDefiner, released error, explicit bool) *capabilityOpening {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	weights := []byte("capability-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	profile := testutil.ArtifactID(t, artifact.KindProfile, "capability-profile")
	model, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "capability/model",
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}, {ID: profile}},
		Manifests: []artifact.Manifest{model},
	}); err != nil {
		t.Fatal(err)
	}
	var requested *artifact.ID
	if explicit {
		missing := testutil.ArtifactID(t, artifact.KindRecipe, "unpublished")
		requested = &missing
	}
	if define != nil {
		definition, err := define(model.ID, profile)
		if err != nil {
			t.Fatal(err)
		}
		if explicit {
			requested = &definition.ID
			if _, _, err := modelrecipe.PublishCandidate(ctx, store, "candidate", definition); err != nil {
				t.Fatal(err)
			}
		} else {
			if err := modelrecipetest.PublishActivation(ctx, store, "capability/active", definition); err != nil {
				t.Fatal(err)
			}
		}
	}
	opening := &capabilityOpening{}
	_, before := store.Head()
	opening.session, opening.err = openCapabilitySession(ctx, store, value,
		mediacapability.Source{Inventory: modelartifact.Inventory{Manifest: model}}, task, requested,
		func(definition recipe.Definition) (evaluation.Runtime, func(context.Context) error, error) {
			if requested != nil && definition.ID != *requested {
				t.Fatal("loader received another recipe")
			}
			opening.loads++
			return &vqaRuntime{}, func(context.Context) error {
				opening.releases++
				return released
			}, nil
		})
	_, after := store.Head()
	opening.wrote = after != before
	if explicit && define != nil {
		status, _, err := modelrecipe.Status(ctx, store, *requested)
		if err != nil || status != recipe.StatusCandidate {
			t.Fatalf("evaluation changed candidate status: %s %v", status, err)
		}
	}
	return opening
}

// capabilityOf is the registered topology of task, as a model's definition.
func capabilityOf(task recipe.Task) capabilityDefiner {
	return func(model, _ artifact.ID) (recipe.Definition, error) {
		return modelrecipe.CapabilityDefinition(task, model)
	}
}

// generatorOf is the generation topology whose first stage is prepare, as a
// model's definition: over the profile when the topology depends on one.
func generatorOf(prepare recipe.ModuleID, profiled bool) capabilityDefiner {
	return func(model, profile artifact.ID) (recipe.Definition, error) {
		if !profiled {
			profile = artifact.ID{}
		}
		return modelrecipe.GenerationDefinition(prepare, model, profile)
	}
}

// hostAutoencoder is the reconstruction topology that runs wholly on the host.
func hostAutoencoder(model, _ artifact.ID) (recipe.Definition, error) {
	return modelrecipe.RepresentationReconstructionDefinition(model)
}

// TestACapabilitySessionNamesTheDeviceItRunsOnAndRefusesOneItCannotSelect holds
// the environment a capability session publishes to where its work is done. A
// recipe with a node off the host runs on a graphics card, so its session names
// that card and not the host; its loader opens the default card only, so a
// manifest that names another is refused before anything is loaded, instead of
// being recorded over work the default card did. A judge is loaded on the card
// the manifest names, so its session names that card. A recipe placed wholly on
// the host still names the host, whatever card the manifest names.
func TestACapabilitySessionNamesTheDeviceItRunsOnAndRefusesOneItCannotSelect(t *testing.T) {
	t.Parallel()
	for name, c := range map[string]struct {
		task    recipe.Task
		define  capabilityDefiner
		ordinal int
		// device and backend are the environment wanted; a refusal leaves them empty.
		device, backend string
	}{
		"a vqa model on the default card":           {recipe.TaskVQA, capabilityOf(recipe.TaskVQA), 0, "cuda:0", "cuda"},
		"a vqa model asked for another card":        {recipe.TaskVQA, capabilityOf(recipe.TaskVQA), 1, "", ""},
		"a card autoencoder on the default card":    {recipe.TaskReconstruction, capabilityOf(recipe.TaskReconstruction), 0, "cuda:0", "cuda"},
		"a card autoencoder asked for another card": {recipe.TaskReconstruction, capabilityOf(recipe.TaskReconstruction), 1, "", ""},
		"the judge of a host image generator":       {recipe.TaskImageGen, generatorOf(modelrecipe.ModuleOscillatorImagePrepare, false), 1, "cuda:1", "cuda"},
		"the judge of a host video generator":       {recipe.TaskVideoGen, generatorOf(modelrecipe.ModuleOscillatorVideoPrepare, false), 1, "cuda:1", "cuda"},
		"the judge of a card image generator":       {recipe.TaskImageGen, generatorOf(modelrecipe.ModuleLatentImagePrepare, true), 1, "cuda:1", "cuda"},
		"a host autoencoder":                        {recipe.TaskReconstruction, hostAutoencoder, 1, "host", "go"},
		"a host image encoder":                      {recipe.TaskImageEmbedding, capabilityOf(recipe.TaskImageEmbedding), 1, "host", "go"},
	} {
		t.Run(name, func(t *testing.T) {
			opening := openPublishedCapability(t, manifest{CodeCommit: hostedTestCommit, Device: c.ordinal}, c.task, c.define, nil, false)
			if c.device == "" {
				if opening.err == nil || opening.session != nil || opening.loads != 0 || opening.releases != 0 || opening.wrote {
					t.Fatalf("device %d opened (%T, %v) with %d loads and %d releases, store written %t, want a refusal before anything is loaded or written",
						c.ordinal, opening.session, opening.err, opening.loads, opening.releases, opening.wrote)
				}
				for _, named := range []string{string(c.task), "device 0", fmt.Sprintf("device %d", c.ordinal)} {
					if !strings.Contains(opening.err.Error(), named) {
						t.Errorf("the refusal %q does not name %q: it names the task, the card that would do the work and the card asked for", opening.err, named)
					}
				}
				return
			}
			if opening.err != nil {
				t.Fatal(opening.err)
			}
			want, err := runrecord.CurrentEnvironment(c.device, c.backend)
			if err != nil {
				t.Fatal(err)
			}
			if got := opening.session.(*capabilitySession).campaign.Authorities().Environment; got != want.ID {
				t.Errorf("the session's environment is %s, want that of %s/%s, %s", got, c.device, c.backend, want.ID)
			}
			if opening.loads != 1 || opening.releases != 0 {
				t.Errorf("the runtime was loaded %d times and released %d times by opening, want once and never", opening.loads, opening.releases)
			}
			if err := opening.session.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestACapabilitySessionReleasesTheRuntimeItLoadedExactlyOnce holds who gives
// the loaded runtime back: a session that opened releases it when it closes and
// reports what the release reported; an opening that fails after loading
// releases it there and reports both failures; and a model with no active
// recipe is refused before anything is loaded.
func TestACapabilitySessionReleasesTheRuntimeItLoadedExactlyOnce(t *testing.T) {
	t.Parallel()
	host, bound := capabilityOf(recipe.TaskImageEmbedding), manifest{CodeCommit: hostedTestCommit}
	stuck := errors.New("the runtime did not release")
	for name, released := range map[string]error{"a release that succeeds": nil, "a release that fails": stuck} {
		t.Run(name, func(t *testing.T) {
			opened := openPublishedCapability(t, bound, recipe.TaskImageEmbedding, host, released, false)
			if opened.err != nil {
				t.Fatal(opened.err)
			}
			if err := opened.session.Close(); !errors.Is(err, released) || opened.loads != 1 || opened.releases != 1 {
				t.Errorf("closing reported %v after %d loads and %d releases, want %v after one of each", err, opened.loads, opened.releases, released)
			}
			unbound := openPublishedCapability(t, manifest{}, recipe.TaskImageEmbedding, host, released, false)
			if unbound.err == nil || unbound.session != nil || released != nil && !errors.Is(unbound.err, released) || unbound.loads != 1 || unbound.releases != 1 {
				t.Errorf("a manifest with no commit opened (%T, %v) with %d loads and %d releases, want a refusal that released the one runtime it loaded",
					unbound.session, unbound.err, unbound.loads, unbound.releases)
			}
		})
	}
	if inactive := openPublishedCapability(t, bound, recipe.TaskImageEmbedding, nil, nil, false); inactive.err == nil || inactive.session != nil || inactive.loads != 0 {
		t.Errorf("a model with no active recipe opened (%T, %v) with %d loads, want a refusal before loading", inactive.session, inactive.err, inactive.loads)
	}
	for name, define := range map[string]capabilityDefiner{"candidate": capabilityOf(recipe.TaskSpeech), "foreign-task": host, "unpublished": nil} {
		t.Run(name, func(t *testing.T) {
			opened := openPublishedCapability(t, bound, recipe.TaskSpeech, define, nil, true)
			if name != "candidate" {
				if opened.err == nil || opened.loads != 0 || opened.wrote {
					t.Fatalf("invalid explicit recipe opened: %+v", opened)
				}
				return
			}
			if opened.err != nil || opened.loads != 1 {
				t.Fatalf("inactive candidate did not open: %+v", opened)
			}
			if err := opened.session.Close(); err != nil || opened.releases != 1 {
				t.Fatalf("candidate release: %v count=%d", err, opened.releases)
			}
		})
	}
}

// TestADirectoryThatIsNoFractaleModelOpensNoThoughtBankSession holds the
// dispatch to what it claims: a path that is not a Fractale generation model
// is reported as not one, with nothing opened, so evaluation falls through to
// the model's own runner.
func TestADirectoryThatIsNoFractaleModelOpensNoThoughtBankSession(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	session, opened, err := openThoughtBankSession(t.Context(), store, manifest{}, t.TempDir())
	if session != nil || opened || err != nil {
		t.Fatalf("an empty directory opened (%v, %t, %v), want no session and no error", session, opened, err)
	}
}

// TestAVQAAnswerIsReplayedOnlyForTheSameQuestionOfTheSameAttempt holds the
// workflow key of a vqa answer to everything the answer depends on: the same
// attempt, image, question and budget keep it, and any change of one moves it,
// so an evaluation run never replays the answers of another.
func TestAVQAAnswerIsReplayedOnlyForTheSameQuestionOfTheSameAttempt(t *testing.T) {
	t.Parallel()
	base, err := vqaRunKey("attempt-1", "a.png", "Describe the image.", 48)
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := vqaRunKey("attempt-1", "a.png", "Describe the image.", 48); again != base {
		t.Fatal("the same attempt, image, question and budget changed the key")
	}
	for name, key := range map[string]func() (string, error){
		"another attempt":  func() (string, error) { return vqaRunKey("attempt-2", "a.png", "Describe the image.", 48) },
		"another image":    func() (string, error) { return vqaRunKey("attempt-1", "b.png", "Describe the image.", 48) },
		"another question": func() (string, error) { return vqaRunKey("attempt-1", "a.png", "Which letter?", 48) },
		"another budget":   func() (string, error) { return vqaRunKey("attempt-1", "a.png", "Describe the image.", 1) },
	} {
		if moved, err := key(); err != nil || moved == base {
			t.Errorf("%s left the key at %q (%v)", name, moved, err)
		}
	}
}

// TestCapabilityTasksAreTheOnesASessionExists holds the manifest's task names
// to the capabilities evaluate opens, and a capability to the runtime it is:
// a forecasting model forecasts no more steps than its horizon and generates
// and scores no text.
func TestCapabilityTasksAreTheOnesASessionExists(t *testing.T) {
	t.Parallel()
	if len(capabilitySessions) == 0 {
		t.Fatal("no capability session is named")
	}
	for _, task := range []string{"seq2seq", "forecast", "vqa", "speech", "image-gen", "video-gen", "image-embedding", "reconstruction"} {
		if capabilitySessions[task] == nil {
			t.Errorf("task %q has no capability session", task)
		}
	}
	speech := speechRuntime{judge: "recognizer"}
	if speech.Judge() != "recognizer" {
		t.Error("a speech session does not name its judge")
	}
	if _, _, err := speech.Generate(t.Context(), "text", inference.GenerateOptions{}); err == nil {
		t.Error("a speech model generated text")
	}
	if _, err := speech.ScoreContinuations(t.Context(), "text", []string{"candidate"}); err == nil {
		t.Error("a speech model scored a continuation")
	}
	if _, err := openSpeechSession(t.Context(), nil, manifest{}, modelRequest{Path: t.TempDir()}); err == nil {
		t.Error("a speech session opened with no judge named")
	}
	embedder := embeddingRuntime{root: t.TempDir()}
	if _, err := embedder.Embed(t.Context(), "../outside.png"); err == nil {
		t.Error("a picture outside the media root was embedded")
	}
	if err := os.WriteFile(filepath.Join(embedder.root, "bad.png"), []byte("not the digest's picture"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := embedder.Embed(t.Context(), "bad.png"); err == nil {
		t.Error("a picture whose bytes are not the digest it is named for was embedded")
	}
	if _, _, err := embedder.Generate(t.Context(), "text", inference.GenerateOptions{}); err == nil {
		t.Error("an image encoder generated text")
	}
	if _, err := openEmbeddingSession(t.Context(), nil, manifest{}, modelRequest{Path: t.TempDir()}); err == nil {
		t.Error("an embedding session opened with no media root")
	}
	vqa := &vqaRuntime{}
	if _, _, err := vqa.Generate(t.Context(), "Which description matches?", inference.GenerateOptions{}); err == nil {
		t.Error("a prompt that names no image was answered")
	}
	if _, _, err := vqa.Generate(t.Context(), "<overgo-media kind=video name=a.mp4>\nWhich?", inference.GenerateOptions{}); err == nil {
		t.Error("a video was answered by a model that reads one image")
	}
	if _, _, err := vqa.Generate(t.Context(), "<overgo-media kind=image name=../a.png>\nWhich?", inference.GenerateOptions{}); err == nil {
		t.Error("an image outside the media root was answered")
	}
	if _, err := vqa.ScoreContinuations(t.Context(), "text", []string{"candidate"}); err == nil {
		t.Error("a vqa evaluation scored a continuation")
	}
	runtime := forecastRuntime{model: &seriesforecast.Model{Dims: seriesforecast.Dims{Horizon: 4, Quantiles: 2}}}
	for _, steps := range []int{0, 5} {
		if _, err := runtime.Forecast([]float32{1}, steps); err == nil {
			t.Errorf("%d steps were forecast by a model whose horizon is 4", steps)
		}
	}
	if _, _, err := runtime.Generate(t.Context(), "text", inference.GenerateOptions{}); err == nil {
		t.Error("a forecasting model generated text")
	}
	if _, err := runtime.ScoreContinuations(t.Context(), "text", []string{"candidate"}); err == nil {
		t.Error("a forecasting model scored a continuation")
	}
}
