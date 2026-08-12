package capabilityruntime

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

var (
	vqaImageContract = artifact.DocumentContract{
		Kind: artifact.KindFile, MediaType: "application/octet-stream", Schema: "overgo.vqa-image-input.v1",
	}
	vqaQuestionContract = artifact.JSONContract(artifact.KindFile, "overgo.vqa-question-input.v1")
	vqaAnswerContract   = artifact.JSONContract(artifact.KindOutput, "overgo.vqa-answer.v1")
)

type VQAImage struct {
	Data []byte
}

type VQAAnswer func(context.Context, VQAImage, string) (string, error)

func ExecuteVQA(
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	key string,
	image VQAImage,
	question string,
	answer VQAAnswer,
) (string, error) {
	if len(image.Data) == 0 || strings.TrimSpace(question) == "" || answer == nil {
		return "", errors.New("capability runtime: VQA image, question, and adapter are required")
	}
	imageID, err := vqaImageContract.Identify(image.Data)
	if err != nil {
		return "", err
	}
	imageContent, err := vqaImageContract.Content(imageID, image.Data)
	if err != nil {
		return "", err
	}
	questionContent, err := artifact.JSONContent(vqaQuestionContract, question)
	if err != nil {
		return "", err
	}
	return Execute[string](ctx, store, modelID, program, key,
		map[recipe.PortName]workflowruntime.Value{
			"image":    workflowruntime.ArtifactValue(recipe.DataImage, image, imageContent),
			"question": workflowruntime.ArtifactValue(recipe.DataText, question, questionContent),
		}, func(runtime *workflowruntime.Runtime) error {
			return runtime.Register(modelrecipe.ModuleVQAAnswer, workflowruntime.AdapterFunc(
				func(ctx context.Context, request workflowruntime.StepRequest) (map[recipe.PortName]workflowruntime.Value, error) {
					if request.Model != modelID {
						return nil, errors.New("capability runtime: VQA program model differs from binding")
					}
					inputImage, err := workflowruntime.ScalarInput[VQAImage](request, "image")
					if err != nil {
						return nil, err
					}
					inputQuestion, err := workflowruntime.ScalarInput[string](request, "question")
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
						"answer": workflowruntime.ArtifactValue(recipe.DataText, value, content),
					}, nil
				},
			))
		})
}
