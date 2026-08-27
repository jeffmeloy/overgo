package main

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

var vqaImageContract = artifact.DocumentContract{
	Kind: artifact.KindFile, MediaType: "application/octet-stream", Schema: "overgo.vqa-image-input.v1",
}
var vqaQuestionContract = artifact.JSONContract(artifact.KindFile, "overgo.vqa-question-input.v1")
var vqaAnswerContract = artifact.JSONContract(artifact.KindOutput, "overgo.vqa-answer.v1")

func resolveActiveVQA(
	ctx context.Context,
	store artifact.Reader,
	modelDir string,
) (artifact.ID, recipe.Program, error) {
	inventory, err := modelartifact.FromHFPath(modelDir)
	if err != nil {
		return artifact.ID{}, recipe.Program{}, err
	}
	modelID := inventory.Manifest.ID
	_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, recipe.TaskVQA)
	return modelID, program, err
}

func vqaInput(definition recipe.Definition, kind recipe.DataKind) (recipe.Input, error) {
	var found recipe.Input
	count := 0
	for _, input := range definition.Inputs {
		if input.Data == kind {
			found, count = input, count+1
		}
	}
	if count != 1 {
		return recipe.Input{}, fmt.Errorf("VQA recipe: need one %s input, got %d", kind, count)
	}
	return found, nil
}

func executeVQA[Prepared any](
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	key string,
	image []byte,
	question string,
	prepare func([]byte, string) (Prepared, error),
	answer func(context.Context, Prepared) (string, error),
) (string, []workflowruntime.NodeWall, error) {
	if len(image) == 0 || strings.TrimSpace(question) == "" || prepare == nil || answer == nil {
		return "", nil, errors.New("VQA recipe: image, question, and stage adapters are required")
	}
	definition := program.Definition()
	imageInput, err := vqaInput(definition, recipe.DataImage)
	if err != nil {
		return "", nil, err
	}
	questionInput, err := vqaInput(definition, recipe.DataText)
	if err != nil {
		return "", nil, err
	}
	stages := program.Stages()
	if len(stages) != 2 || len(stages[0].Module.Outputs) != 1 || len(stages[1].Module.Inputs) != 1 {
		return "", nil, fmt.Errorf("VQA recipe: need prepare and generate stages, got %d", len(stages))
	}
	imageContent, err := vqaImageContract.ContentBytes(image)
	if err != nil {
		return "", nil, err
	}
	questionContent, err := artifact.JSONContent(vqaQuestionContract, question)
	if err != nil {
		return "", nil, err
	}
	return capabilityruntime.ExecuteMeasured[string](ctx, store, modelID, program, key,
		map[recipe.PortName]workflowruntime.Value{
			imageInput.Name:    workflowruntime.ArtifactValue(imageInput.Data, image, imageContent),
			questionInput.Name: workflowruntime.ArtifactValue(questionInput.Data, question, questionContent),
		}, func(runtime *workflowruntime.Runtime) error {
			if err := workflowruntime.RegisterResolvedStage(
				runtime, stages[0].Module.ID, modelID,
				func(_ context.Context, request workflowruntime.StepRequest) (Prepared, error) {
					inputImage, err := workflowruntime.ScalarInput[[]byte](request, imageInput.Target.Port)
					if err != nil {
						var zero Prepared
						return zero, err
					}
					inputQuestion, err := workflowruntime.ScalarInput[string](request, questionInput.Target.Port)
					if err != nil {
						var zero Prepared
						return zero, err
					}
					return prepare(inputImage, inputQuestion)
				}, nil,
			); err != nil {
				return err
			}
			return workflowruntime.RegisterResolvedStage(
				runtime, stages[1].Module.ID, modelID,
				func(ctx context.Context, request workflowruntime.StepRequest) (string, error) {
					prepared, err := workflowruntime.ScalarInput[Prepared](request, stages[1].Module.Inputs[0].Name)
					if err != nil {
						return "", err
					}
					return answer(ctx, prepared)
				}, func(value string) (artifact.Content, error) {
					return artifact.JSONContent(vqaAnswerContract, value)
				},
			)
		})
}
