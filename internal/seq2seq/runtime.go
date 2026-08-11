package seq2seq

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var generatedTokensContract = artifact.JSONContract(artifact.KindFile, "overgo.seq2seq-tokens.v1")

type GenerateRequest struct {
	Source    []int `json:"source"`
	MaxTokens int   `json:"max_tokens"`
}

func ValidateGenerateRequest(request GenerateRequest) error {
	if len(request.Source) == 0 || request.MaxTokens <= 0 {
		return errors.New("seq2seq: generation requires source tokens and a positive token limit")
	}
	return nil
}

type generator interface {
	Generate([]int, int) ([]int, error)
}

// RegisterRuntime binds one loaded model to its generation stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("seq2seq: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, model)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model generator) error {
	if model == nil {
		return errors.New("seq2seq: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONStage[GenerateRequest, []int](
		runtime, modelrecipe.ModuleSeq2SeqGenerate, modelID, generatedTokensContract,
		func(request GenerateRequest) ([]int, error) {
			if err := ValidateGenerateRequest(request); err != nil {
				return nil, err
			}
			return model.Generate(request.Source, request.MaxTokens)
		},
	)
}
