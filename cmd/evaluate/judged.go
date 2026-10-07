package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/tokenizer"
)

// judgedRuntime is a vision-language model that answers questions about the
// pictures another model drew, as the judge of that model: before it looks at a
// picture it checks that the model and recipe being evaluated drew it.
type judgedRuntime struct {
	projectionRuntime
	judge  string
	model  artifact.ID
	recipe artifact.ID
}

// Judge names the model that answers.
func (r judgedRuntime) Judge() string { return r.judge }

// Generate answers the prompt's question about its picture once the picture is
// shown to be one the evaluated recipe drew.
func (r judgedRuntime) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	_, name, _, named := evaluation.ParseMediaPrompt(prompt)
	if !named {
		return nil, "", errors.New("evaluate: a judged case names the picture its question asks about")
	}
	if err := verifyRendered(r.root, r.model, r.recipe, name); err != nil {
		return nil, "", err
	}
	return r.projectionRuntime.Generate(ctx, prompt, options)
}

// judgedSession is the opener of a generation task's judged session, for the
// table of capability sessions by task.
func judgedSession(task recipe.Task) func(context.Context, *overgodb.Store, manifest, modelRequest) (evaluationSession, error) {
	return func(ctx context.Context, store *overgodb.Store, value manifest, request modelRequest) (evaluationSession, error) {
		return openJudgedSession(ctx, store, value, request, task)
	}
}

// openJudgedSession opens the image or video generation model at the request's path
// under its active recipe for task, judged by the vision-language model the
// request names, which looks at the media of its media root: each must be one
// the recipe drew, by the root's render manifest.
func openJudgedSession(ctx context.Context, store *overgodb.Store, value manifest, request modelRequest, task recipe.Task) (evaluationSession, error) {
	if request.Judge == "" || request.Projector == "" || request.MediaRoot == "" {
		return nil, fmt.Errorf("evaluate: %s is a %s model, which a vision-language model judges, and the request names no judge, projector or media root", request.Path, task)
	}
	source, err := mediacapability.Catalog[task].Resolve(request.Path)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no %s model: %w", request.Path, task, err)
	}
	modelID := source.Inventory.Manifest.ID
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, task)
	if err != nil || !active {
		return nil, errors.Join(err, fmt.Errorf("evaluate: %s has no active %s recipe", request.Path, task))
	}
	opened, err := openCapabilitySession(ctx, store, value, source, task, func() (evaluation.Runtime, func(context.Context) error, error) {
		runner, err := modelcli.OpenRunner(ctx, value.Repository, request.Judge, inference.OpenOptions{DeviceOrdinal: value.Device})
		if err != nil {
			return nil, nil, err
		}
		vision, err := projector.OpenActiveSession(ctx, store, runner.ModelID(), request.Projector, projector.OpenOptions{CUDA: true, DeviceOrdinal: value.Device})
		if err != nil {
			return nil, nil, errors.Join(err, runner.Close())
		}
		return judgedRuntime{
				projectionRuntime: projectionRuntime{runner: runner, vision: vision, root: request.MediaRoot, ffmpeg: os.Getenv("OVERGO_FFMPEG")},
				judge:             filepath.Base(request.Judge), model: modelID, recipe: activation.Definition.ID,
			}, func(context.Context) error {
				return errors.Join(vision.Close(), runner.Close())
			}, nil
	})
	if err != nil {
		return nil, err
	}
	opened.(*capabilitySession).campaign.WithPrompting(evaluation.PromptingChatTemplate)
	return opened, nil
}
