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

func executeVQA(
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	key string,
	image []byte,
	question string,
	answer func(context.Context, []byte, string) (string, error),
) (string, error) {
	if len(image) == 0 || strings.TrimSpace(question) == "" || answer == nil {
		return "", errors.New("VQA recipe: image, question, and adapter are required")
	}
	definition := program.Definition()
	imageInput, err := vqaInput(definition, recipe.DataImage)
	if err != nil {
		return "", err
	}
	questionInput, err := vqaInput(definition, recipe.DataText)
	if err != nil {
		return "", err
	}
	stages := program.Stages()
	if len(stages) != 1 {
		return "", fmt.Errorf("VQA recipe: need one adapter stage, got %d", len(stages))
	}
	imageContent, err := vqaImageContract.ContentBytes(image)
	if err != nil {
		return "", err
	}
	questionContent, err := artifact.JSONContent(vqaQuestionContract, question)
	if err != nil {
		return "", err
	}
	return capabilityruntime.Execute[string](ctx, store, modelID, program, key,
		map[recipe.PortName]workflowruntime.Value{
			imageInput.Name:    workflowruntime.ArtifactValue(imageInput.Data, image, imageContent),
			questionInput.Name: workflowruntime.ArtifactValue(questionInput.Data, question, questionContent),
		}, func(runtime *workflowruntime.Runtime) error {
			return runtime.Register(stages[0].Module.ID, workflowruntime.AdapterFunc(
				func(ctx context.Context, request workflowruntime.StepRequest) (map[recipe.PortName]workflowruntime.Value, error) {
					inputImage, err := workflowruntime.ScalarInput[[]byte](request, imageInput.Name)
					if err != nil {
						return nil, err
					}
					inputQuestion, err := workflowruntime.ScalarInput[string](request, questionInput.Name)
					if err != nil {
						return nil, err
					}
					value, err := answer(ctx, inputImage, inputQuestion)
					if err != nil {
						return nil, err
					}
					content, err := artifact.JSONContent(vqaAnswerContract, value)
					if err != nil {
						return nil, err
					}
					return map[recipe.PortName]workflowruntime.Value{
						definition.Outputs[0].Source.Port: workflowruntime.ArtifactValue(recipe.DataText, value, content),
					}, nil
				},
			))
		})
}
