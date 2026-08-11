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
	EncodeRequest(GenerateRequest) (EncodedRequest, error)
	PrepareGeneration(EncodedRequest) (TokenSelector, error)
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
		runtime, modelrecipe.ModuleSeq2SeqEncode, modelID, model.EncodeRequest, nil,
	); err != nil {
		return err
	}
	if err := workflowruntime.RegisterScalarStage(
		runtime, modelrecipe.ModuleSeq2SeqPrepare, modelID, model.PrepareGeneration, nil,
	); err != nil {
		return err
	}
	return workflowruntime.RegisterJSONStage[TokenSelector, []int](
		runtime, modelrecipe.ModuleSeq2SeqSelect, modelID, generatedTokensContract,
		func(selector TokenSelector) ([]int, error) {
			if selector == nil {
				return nil, errors.New("seq2seq: missing token selector")
			}
			return selector.Select()
		},
	)
}
