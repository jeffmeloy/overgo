package visionqa

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

var answerContract = artifact.JSONContract(artifact.KindOutput, "overgo.vqa-answer.v1")

type Image struct {
	Data []byte `json:"data"`
}

func ValidateInput(image Image, question string) error {
	if len(image.Data) == 0 || strings.TrimSpace(question) == "" {
		return errors.New("visionqa: image and question are required")
	}
	return nil
}

func RegisterRuntime(
	runtime *workflowruntime.Runtime,
	modelID artifact.ID,
	answer func(context.Context, Image, string) (string, error),
) error {
	if answer == nil {
		return errors.New("visionqa: incomplete runtime binding")
	}
	return runtime.Register(modelrecipe.ModuleVQAAnswer, workflowruntime.AdapterFunc(
		func(ctx context.Context, request workflowruntime.StepRequest) (map[recipe.PortName]workflowruntime.Value, error) {
			if request.Model != modelID {
				return nil, errors.New("visionqa: recipe model differs from runtime binding")
			}
			image, err := workflowruntime.ScalarInput[Image](request, "image")
			if err != nil {
				return nil, err
			}
			question, err := workflowruntime.ScalarInput[string](request, "question")
			if err != nil {
				return nil, err
			}
			if err := ValidateInput(image, question); err != nil {
				return nil, err
			}
			value, err := answer(ctx, image, question)
			if err != nil {
				return nil, err
			}
			content, err := artifact.JSONContent(answerContract, value)
			if err != nil {
				return nil, err
			}
			return map[recipe.PortName]workflowruntime.Value{
				"answer": workflowruntime.ArtifactValue(recipe.DataText, value, content),
			}, nil
		},
	))
}
