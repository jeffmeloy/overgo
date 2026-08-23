package thoughtbank

import (
	"context"
	"errors"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/hfbpe"
	"overgo/internal/modelrecipe"
	"overgo/internal/textgeneration"
	"overgo/internal/workflowruntime"
)

var generationContract = artifact.JSONContract(artifact.KindOutput, "overgo.thoughtbank-generation.v1")

type Generation struct {
	Text   string `json:"text"`
	Tokens []int  `json:"tokens"`
}

type Generator struct {
	weights   *FastWeightBankLMWeights
	config    ArchConfig
	tokenizer *hfbpe.Tokenizer
}

func LoadGenerator(directory string) (*Generator, error) {
	weights, config, err := LoadCheckpoint(filepath.Join(directory, "model.pt"))
	if err != nil {
		return nil, err
	}
	tokenizer, err := hfbpe.Load(directory)
	if err != nil {
		return nil, err
	}
	return &Generator{weights: weights, config: config, tokenizer: tokenizer}, nil
}

func (g *Generator) Close(context.Context) error {
	if g != nil {
		g.weights, g.tokenizer = nil, nil
	}
	return nil
}

func (g *Generator) Generate(request textgeneration.Request) (Generation, error) {
	if g == nil || g.weights == nil || g.tokenizer == nil {
		return Generation{}, errors.New("thoughtbank: generator is unavailable")
	}
	if err := textgeneration.Validate(request); err != nil {
		return Generation{}, err
	}
	encoded, err := g.tokenizer.Encode(request.Text)
	if err != nil {
		return Generation{}, err
	}
	if len(encoded) == 0 {
		return Generation{}, errors.New("thoughtbank: prompt tokenization is empty")
	}
	prompt := make([]int32, len(encoded))
	for index, token := range encoded {
		prompt[index] = int32(token)
	}
	slots := g.config.MemSeedSlots
	state, logits, err := FastWeightBankLMDecodeInit(
		g.weights, prompt, make([]float32, slots*g.config.MemDim), slots,
	)
	if err != nil {
		return Generation{}, err
	}
	tokens := make([]int, request.MaxTokens)
	for index := range tokens {
		tokens[index] = argmaxLogit(logits)
		if index+1 < len(tokens) {
			logits, err = FastWeightBankLMDecodeStep(state, int32(tokens[index]))
			if err != nil {
				return Generation{}, err
			}
		}
	}
	return Generation{Text: g.tokenizer.Decode(tokens), Tokens: tokens}, nil
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generator *Generator) error {
	if generator == nil {
		return errors.New("thoughtbank: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONStage[textgeneration.Request, Generation](
		runtime, modelrecipe.ModuleThoughtBankGenerate, modelID, generationContract, generator.Generate,
	)
}

func argmaxLogit(values []float32) int {
	best := 0
	for index := range values {
		if values[index] > values[best] {
			best = index
		}
	}
	return best
}
