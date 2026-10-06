package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// projectionRuntime is a model and its multimodal projector as an evaluation
// runtime: a case whose prompt names a video of the media root is asked of the
// model with that video's frames projected into its prompt, and answers
// generatively; the projection recipe frames the conversation itself, so the
// prompt is not shaped again.
type projectionRuntime struct {
	runner *inference.Runner
	vision projector.Session
	root   string
	ffmpeg string
}

// ShapeChatPrompt returns the prompt itself: the projector shapes it.
func (projectionRuntime) ShapeChatPrompt(prompt string) (string, error) { return prompt, nil }

// ScoreContinuations refuses: a projection suite scores generated answers.
func (projectionRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("evaluate: a projection evaluation scores generated answers, not continuation likelihoods")
}

// Generate asks the question of the video the prompt names.
func (r projectionRuntime) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	kind, name, question, named := evaluation.ParseMediaPrompt(prompt)
	if !named || kind != evaluation.MediaVideo {
		return nil, "", errors.New("evaluate: a projection case names the video its question asks about")
	}
	path, err := mediaPath(r.root, name)
	if err != nil {
		return nil, "", err
	}
	video := r.runner.RuntimePolicy().Interactive.Video
	frames, err := media.DecodeVideoFile(ctx, path, r.ffmpeg, video.FPS, video.MaxFrames)
	if err != nil {
		return nil, "", fmt.Errorf("evaluate: decode video %s: %w", name, err)
	}
	built, err := r.vision.BuildVideoPrompt(ctx, r.runner, frames, "", question, video.FPS, false)
	if err != nil {
		return nil, "", fmt.Errorf("evaluate: project video %s: %w", name, err)
	}
	promptIDs, projected, err := inference.CompileProjectedInputs(built, r.runner.Spec().EmbeddingLength)
	if err != nil {
		return nil, "", err
	}
	options.PromptTokenIDs, options.ProjectedInputs = promptIDs, &projected
	return r.runner.Generate(ctx, question, options)
}

// defaultCaptionCases and defaultCaptionSeed are the clips a caption-matching
// suite asks about, and the seed that picks them, unless a run names others: the
// suite the scorecard's held-out quality reads.
const (
	defaultCaptionCases = 64
	defaultCaptionSeed  = 17
)

// captionFile is the caption file a VidGen clip directory holds beside its
// clips.
const captionFile = "VidGen_1M_video_caption.json"

// writeCaptionMatching builds the video caption-matching suite of a clip
// directory and writes it to out, the suite file a projection evaluation runs.
// With a frames directory it builds the suite of the clips' middle frames, for
// a model that reads one image and answers in a description, and cuts those
// frames to the directory.
func writeCaptionMatching(ctx context.Context, clips, captions, out, frames string, cases int, seed uint64) error {
	if captions == "" {
		captions = filepath.Join(clips, captionFile)
	}
	spec := evaluation.CaptionMatchingSpec{Clips: clips, Captions: captions, Cases: cases, Seed: seed}
	if frames != "" {
		suite, err := evaluation.BuildFrameDescriptionMatching(spec)
		if err != nil {
			return err
		}
		if err := writeFrames(ctx, suite, clips, frames, os.Getenv("OVERGO_FFMPEG")); err != nil {
			return err
		}
		return writeSuite(out, suite, len(suite.Cases), "frame description-matching")
	}
	suite, err := evaluation.BuildCaptionMatching(spec)
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "caption-matching")
}

// mediaPath resolves a case's media name inside the media root: a name that
// climbs out of the root names no media of it.
func mediaPath(root, name string) (string, error) {
	if root == "" || !filepath.IsLocal(name) {
		return "", fmt.Errorf("evaluate: media %q is outside the media root %q", name, root)
	}
	path := filepath.Join(root, name)
	if _, err := os.Stat(path); err != nil {
		return "", err
	}
	return path, nil
}

// projectionSession evaluates a model under its active projection recipe: the
// records it publishes bind that recipe, so the scorecard reads them as the
// projection recipe's held-out quality.
type projectionSession struct {
	nativeSession
	vision projector.Session
}

// openProjectionSession opens the model and the projector the request names,
// and binds the campaign to the model's active projection recipe over its
// definition.
func openProjectionSession(ctx context.Context, store *overgodb.Store, value manifest, request modelRequest) (evaluationSession, error) {
	runner, err := modelcli.OpenRunner(ctx, value.Repository, request.Path, inference.OpenOptions{DeviceOrdinal: value.Device})
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (evaluationSession, error) { return nil, errors.Join(cause, runner.Close()) }
	description, err := runner.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil {
		return fail(err)
	}
	identity := description.Identity
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, identity.Model, recipe.TaskProjection)
	if err != nil || !active {
		return fail(errors.Join(err, fmt.Errorf("evaluate: %s has no active projection recipe", request.Path)))
	}
	identity.Recipe, identity.RecipeVersion = activation.Definition.ID, activation.Definition.Version
	environment, err := runrecord.CurrentEnvironment(fmt.Sprintf("cuda:%d", value.Device), "cuda")
	if err != nil {
		return fail(err)
	}
	vision, err := projector.OpenActiveSession(ctx, store, runner.ModelID(), request.Projector, projector.OpenOptions{CUDA: true, DeviceOrdinal: value.Device})
	if err != nil {
		return fail(err)
	}
	runtime := projectionRuntime{runner: runner, vision: vision, root: request.MediaRoot, ffmpeg: os.Getenv("OVERGO_FFMPEG")}
	campaign, err := evaluation.NewIsolatedCampaign(store, runtime, identity, environment, value.CodeCommit)
	if err != nil {
		return fail(errors.Join(err, vision.Close()))
	}
	campaign.WithPrompting(evaluation.PromptingChatTemplate)
	return &projectionSession{
		nativeSession: nativeSession{store: store, runner: runner, campaign: campaign, model: identity.Model, commit: value.CodeCommit},
		vision:        vision,
	}, nil
}

// Close releases the projector, then the model and the store.
func (s *projectionSession) Close() error {
	if s == nil {
		return nil
	}
	return errors.Join(s.vision.Close(), s.nativeSession.Close())
}
