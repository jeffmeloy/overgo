package seq2seq

import (
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/tokenizer"
	"overgo/internal/workflowruntime"
)

var generatedTextContract = artifact.JSONContract(artifact.KindFile, "overgo.seq2seq-text.v1")

type GenerateRequest struct {
	Text      string `json:"text"`
	MaxTokens int    `json:"max_tokens"`
}

func ValidateGenerateRequest(request GenerateRequest) error {
	if request.Text == "" || request.MaxTokens <= 0 {
		return errors.New("seq2seq: generation requires text and a positive token limit")
	}
	return nil
}

type runtimeStages interface {
	encodeRequest(GenerateRequest) (encodedRequest, error)
	prepareGeneration(encodedRequest) (textSelector, error)
}

// Generator binds one model to its artifact tokenizer.
type Generator struct {
	model     *Model
	tokenizer *tokenizer.Unigram
}

func LoadGenerator(directory string) (*Generator, error) {
	model, err := Load(directory)
	if err != nil {
		return nil, err
	}
	pieces, unknown, err := tokenizer.ReadSentencePieceModel(filepath.Join(directory, "tokenizer.model"))
	if err != nil {
		return nil, fmt.Errorf("seq2seq: load tokenizer: %w", err)
	}
	table, err := tokenizer.NewUnigram(pieces, unknown)
	if err != nil {
		return nil, err
	}
	return &Generator{model: model, tokenizer: table}, nil
}

func (g *Generator) Generate(request GenerateRequest) (string, error) {
	encoded, err := g.encodeRequest(request)
	if err != nil {
		return "", err
	}
	selector, err := g.prepareGeneration(encoded)
	if err != nil {
		return "", err
	}
	return selector.selectText()
}

func (g *Generator) encodeRequest(request GenerateRequest) (encodedRequest, error) {
	if g == nil || g.model == nil || g.tokenizer == nil {
		return encodedRequest{}, errors.New("seq2seq: generator is unavailable")
	}
	if err := ValidateGenerateRequest(request); err != nil {
		return encodedRequest{}, err
	}
	source, err := g.tokenizer.Encode(request.Text)
	if err != nil {
		return encodedRequest{}, err
	}
	return g.model.encodeTokens(source, request.MaxTokens)
}

func (g *Generator) prepareGeneration(encoded encodedRequest) (textSelector, error) {
	selector, err := g.model.prepareTokens(encoded)
	if err != nil {
		return nil, err
	}
	return textGeneration{tokens: selector, tokenizer: g.tokenizer}, nil
}

// RegisterRuntime binds one loaded generator to its text stages.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generator *Generator) error {
	if generator == nil {
		return errors.New("seq2seq: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, generator)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model runtimeStages) error {
	if model == nil {
		return errors.New("seq2seq: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONPipeline(
		runtime, modelID, generatedTextContract,
		modelrecipe.ModuleSeq2SeqEncode, model.encodeRequest,
		modelrecipe.ModuleSeq2SeqPrepare, model.prepareGeneration,
		modelrecipe.ModuleSeq2SeqSelect,
		func(selector textSelector) (string, error) {
			if selector == nil {
				return "", errors.New("seq2seq: missing text selector")
			}
			return selector.selectText()
		},
	)
}
