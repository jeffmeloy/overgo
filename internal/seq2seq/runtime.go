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

type runtimeStages interface {
	encodeRequest(GenerateRequest) (encodedRequest, error)
	prepareGeneration(encodedRequest) (tokenSelector, error)
}

// RegisterRuntime binds one loaded model to its generation stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *Model) error {
	if model == nil {
		return errors.New("seq2seq: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, model)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model runtimeStages) error {
	if model == nil {
		return errors.New("seq2seq: incomplete runtime binding")
	}
	if err := workflowruntime.RegisterScalarStage(
		runtime, modelrecipe.ModuleSeq2SeqEncode, modelID, model.encodeRequest, nil,
	); err != nil {
		return err
	}
	if err := workflowruntime.RegisterScalarStage(
		runtime, modelrecipe.ModuleSeq2SeqPrepare, modelID, model.prepareGeneration, nil,
	); err != nil {
		return err
	}
	return workflowruntime.RegisterJSONStage[tokenSelector, []int](
		runtime, modelrecipe.ModuleSeq2SeqSelect, modelID, generatedTokensContract,
		func(selector tokenSelector) ([]int, error) {
			if selector == nil {
				return nil, errors.New("seq2seq: missing token selector")
			}
			return selector.selectTokens()
		},
	)
}
