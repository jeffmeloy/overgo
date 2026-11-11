package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/envsetting"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/tokenizer"
)

// judgedRuntime is a vision-language model that answers questions about the
// pictures another model drew, as the judge of that model: before it looks at a
// picture it checks that the model and recipe being evaluated drew it.
type judgedRuntime struct {
	projectionRuntime
	model  artifact.ID
	recipe artifact.ID
	store  artifact.Reader
}

// Judge names the model that answers.
func (r judgedRuntime) Judge() string {
	id, _ := artifact.JSONID(artifact.KindProfile, r.identity)
	return id.String()
}

// Generate answers the prompt's question about its picture once the picture is
// shown to be one the evaluated recipe drew.
func (r judgedRuntime) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	_, name, _, named := evaluation.ParseMediaPrompt(prompt)
	if !named {
		return nil, "", errors.New("evaluate: a judged case names the picture its question asks about")
	}
	if r.mediaFiles == nil {
		return nil, "", errors.New("evaluate: media was not bound")
	}
	bound, found := r.files[name]
	if !found || !bound.input.Producer.Valid() {
		return nil, "", errors.New("evaluate: rendered provenance was not bound")
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
	source, err := mediacapability.Catalog[task].Resolve(ctx, request.Path)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no %s model: %w", request.Path, task, err)
	}
	modelID := source.Inventory.Manifest.ID
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, task)
	if err != nil || !active {
		return nil, errors.Join(err, fmt.Errorf("evaluate: %s has no active %s recipe", request.Path, task))
	}
	opened, err := openCapabilitySession(ctx, store, value, source, task, nil, func(recipe.Definition) (evaluation.Runtime, func(context.Context) error, error) {
		// The judge's projector opens first: the judge's expert cache takes the device memory it leaves.
		judge, _, err := modelcli.ResolveModel(ctx, store, request.Judge)
		if err != nil {
			return nil, nil, err
		}
		// The judge answers only on current evidence, which the parent re-verified where it could (currentJudges).
		if err := currentJudges(ctx, []modelRequest{request}, func(ctx context.Context, _ string) (modelrecipe.ModelState, error) {
			_, state, err := modelrecipe.ReadModelState(ctx, store, judge, recipe.TaskInference)
			return state, err
		}, func(context.Context, string) error {
			return errors.New("a worker re-verifies no judge: the run's parent does, before any worker opens one")
		}); err != nil {
			return nil, nil, err
		}
		reserve, bind := modelcli.ModelReserve(ctx)
		vision, err := projector.OpenActiveSession(ctx, store, judge, request.Projector, projector.OpenOptions{CUDA: true, DeviceOrdinal: value.Device, MemoryReserve: reserve})
		if err != nil {
			return nil, nil, err
		}
		runner, err := modelcli.OpenRunner(ctx, value.Repository, request.Judge, inference.OpenOptions{DeviceOrdinal: value.Device})
		if err != nil {
			return nil, nil, errors.Join(err, vision.Close())
		}
		bind(runner)
		runtime := projectionRuntime{runner: runner, vision: vision, mediaFiles: &mediaFiles{root: request.MediaRoot}, ffmpeg: envsetting.Result("OVERGO_FFMPEG")}
		runtime.identity, err = projectionIdentity(ctx, store, runtime)
		if err != nil {
			return nil, nil, errors.Join(err, vision.Close(), runner.Close())
		}
		return judgedRuntime{
				projectionRuntime: runtime,
				model:             modelID, recipe: activation.Definition.ID, store: store,
			}, func(context.Context) error {
				return errors.Join(vision.Close(), runner.Close())
			}, nil
	})
	if err != nil {
		return nil, err
	}
	keepJudgedAnswers(opened.(*capabilitySession).campaign)
	return opened, nil
}

// keepJudgedAnswers frames a judged session's questions in the judge's conversation and keeps the judge's answers
// under the identity of its code where the build counted one (evaluation.AnsweredBy), else under the commit: a judged
// question costs 6 to 27 s of the card, and a later commit that leaves the judge's code where it was asks it nothing it
// answered before about the same media.
func keepJudgedAnswers(campaign *evaluation.Campaign) *evaluation.Campaign {
	return evaluation.AnsweredBy[inference.Runner](campaign.WithPrompting(evaluation.PromptingChatTemplate))
}

// currentJudges re-verifies, before any judged session opens, each vision-language judge the models name whose evidence
// is out of date (recipe verify -recorded), and refuses one whose verification is not current after it: a score from a
// judge whose verification the build no longer holds is bound to no verified build (SQA-315), and the Wan student's
// 0.898 on 2026-11-08 was judged after the note that said so. A judge named twice is read once.
func currentJudges(ctx context.Context, models []modelRequest, state func(context.Context, string) (modelrecipe.ModelState, error), reverify func(context.Context, string) error) error {
	var read []string
	for _, request := range models {
		if request.Judge == "" || request.Projector == "" || slices.Contains(read, request.Judge) {
			continue
		}
		read = append(read, request.Judge)
		current, err := state(ctx, request.Judge)
		if err == nil && current.Verification == modelrecipe.StateOutOfDate {
			fmt.Fprintf(os.Stderr, "evaluate: the judge %s holds evidence out of date; re-verifying it on its recorded suite\n", request.Judge)
			if err = reverify(ctx, request.Judge); err == nil {
				current, err = state(ctx, request.Judge)
			}
		}
		if err != nil {
			return fmt.Errorf("evaluate: the judge %s: %w", request.Judge, err)
		}
		if current.Verification != modelrecipe.StateVerified {
			return fmt.Errorf("evaluate: the judge %s is %s (%s); a judged run opens its judge only with current verification evidence: go run ./cmd/recipe verify -task inference -recorded %s",
				request.Judge, current.Verification, current.Reason, request.Judge)
		}
	}
	return nil
}

// judgeState reads a judge's verification in the store at repository, without loading its weights.
func judgeState(repository string) func(context.Context, string) (modelrecipe.ModelState, error) {
	return func(ctx context.Context, path string) (modelrecipe.ModelState, error) {
		store, err := overgodb.OpenReadOnly(repository)
		if err != nil {
			return modelrecipe.ModelState{}, err
		}
		state, err := modelcli.CurrentState(ctx, store, path, recipe.TaskInference)
		return state, errors.Join(err, store.Close())
	}
}

// reverifyJudge re-verifies a judge into the store at repository on the suite its accepted verification ran.
func reverifyJudge(repository string) func(context.Context, string) error {
	return func(ctx context.Context, path string) error {
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{
			Path: "go", Args: []string{"run", "./cmd/recipe", "verify", "-task", string(recipe.TaskInference), "-recorded", "-repo", repository, path},
			Stdout: os.Stderr, Stderr: os.Stderr,
		})
		if err == nil && receipt.ExitCode != 0 {
			err = fmt.Errorf("recipe verify -recorded exited with status %d", receipt.ExitCode)
		}
		return err
	}
}
