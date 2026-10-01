package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// TextHeldoutRequest trains a text model under a registered token-prediction
// objective and judges it on the objective's held-out split.
type TextHeldoutRequest struct {
	Alias          string
	ModelDirectory string
	Steps          int
	// MaximumSequence bounds each record's tokens; zero uses the model's
	// context length.
	MaximumSequence int
	Host            bool
	// Device trains the model when Host is false.
	Device           DeviceTrainer
	MaxProjectedWall time.Duration
}

// promptAnswers is one record of a generation objective: a prompt and the
// answers a generated continuation is accepted as. Training reads the
// prompt followed by the first answer.
type promptAnswers struct {
	Prompt  string   `json:"prompt"`
	Answers []string `json:"answers"`
}

// generationPrompt is what a held-out generation answers from: the
// prompt's tokens, and how many tokens to generate, as many as the longest
// accepted answer takes after the prompt, so length is not what is judged.
type generationPrompt struct {
	tokens []int
	length int
}

// RunTextHeldout trains the model on the training records' token sequences
// in turn and judges it on the held-out records by the objective's metric.
// Token accuracy reads each record as one UTF-8 document and asks, at each
// position of a held-out record, whether the model's most likely next token
// is the one that follows. Generated answer reads each record as a JSON
// prompt and its accepted answers, trains on the prompt and its first
// answer, and asks whether the text the model generates greedily after a
// held-out prompt is an accepted answer. The model's weights update in
// place, so the base and trained models answer in one process.
func RunTextHeldout(ctx context.Context, store *overgodb.Store, request TextHeldoutRequest) (HeldoutResult, error) {
	objective, err := LoadObjectiveAlias(ctx, store, request.Alias)
	if err != nil {
		return HeldoutResult{}, err
	}
	generation := objective.Metric == trainingprogram.MetricGeneratedAnswer
	if objective.Kind != trainingprogram.ObjectiveTokenPrediction || objective.Metric != trainingprogram.MetricTokenAccuracy && !generation {
		return HeldoutResult{}, fmt.Errorf("training workflow: objective %q is %q judged by %q, not token prediction by token accuracy or generated answer", request.Alias, objective.Kind, objective.Metric)
	}
	model, err := loadTrainableModel(request.ModelDirectory)
	if err != nil {
		return HeldoutResult{}, err
	}
	tokenizer, err := loadTrainableTokenizer(request.ModelDirectory)
	if err != nil {
		return HeldoutResult{}, err
	}
	weights, err := IdentifyModelWeights(request.ModelDirectory)
	if err != nil {
		return HeldoutResult{}, err
	}
	sequence := model.Dims.ContextLength
	if request.MaximumSequence > 0 {
		sequence = min(sequence, request.MaximumSequence)
	}
	prompts := map[string]generationPrompt{}
	// Each record is one UTF-8 document; its example is its first sequence
	// tokens: the input is the sequence, the target every token after the
	// first. A generation record's document is its prompt and answers, and
	// its sequence the prompt followed by the first answer.
	examples, _, err := ObjectiveExamples(ctx, store, objective, []byte("overgo/training/text-utf8/v1"), recipecontract.ModalityText,
		trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "utf-8"),
		func(example trainingdata.Example) (HeldoutExample, error) {
			if len(example.Values) != 1 {
				return HeldoutExample{}, errors.New("not one text document")
			}
			text, answers := string(example.Values[0].Data), []string(nil)
			if generation {
				record, prompt, err := readPromptAnswers(tokenizer.encode, example.Values[0].Data)
				if err != nil {
					return HeldoutExample{}, err
				}
				prompts[example.ID] = prompt
				text, answers = record.Prompt+record.Answers[0], record.Answers
			}
			tokens, err := tokenizer.encode(text)
			if err != nil {
				return HeldoutExample{}, err
			}
			tokens = tokens[:min(len(tokens), sequence)]
			if len(tokens) < 2 {
				return HeldoutExample{}, fmt.Errorf("encodes to %d token(s); a next-token target needs two", len(tokens))
			}
			return HeldoutExample{Input: dtype.ConvertSlice[float64](tokens), Target: dtype.ConvertSlice[float64](tokens[1:]), Answers: answers}, nil
		})
	if err != nil {
		return HeldoutResult{}, err
	}
	muonPlan, err := model.TrainingPlan()
	if err != nil {
		return HeldoutResult{}, err
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction, Updates: request.Steps, MaximumSequence: sequence,
		MaxProjectedWall: request.MaxProjectedWall,
		Parameters:       muonPlan.ParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
	if err != nil {
		return HeldoutResult{}, err
	}
	config := plan.Optimizer()
	return RunHeldout(ctx, store, HeldoutRun{
		Alias: request.Alias, ModelPath: weights, Examples: examples, Host: request.Host,
		Train: func(_ context.Context, _ artifact.ID, shown []HeldoutExample) error {
			if len(shown) == 0 {
				return errors.New("training workflow: the training membership holds no text")
			}
			batches := make([][]int, plan.Updates())
			for step := range batches {
				batches[step] = dtype.ConvertSlice[int](shown[step%len(shown)].Input)
			}
			_, _, _, err := runTrainingState(model, batches, config.BaseLearningRate, config.Momentum, request.Device, !request.Host, false, nil, nil, nil)
			return err
		},
		Predict: func(example HeldoutExample) (HeldoutPrediction, error) {
			if generation {
				return generateAnswer(model, tokenizer.piece, prompts[example.Record])
			}
			return nextTokens(model, example)
		},
	})
}

// readPromptAnswers reads one generation record and what a held-out
// generation answers it from.
func readPromptAnswers(encode func(string) ([]int, error), data []byte) (promptAnswers, generationPrompt, error) {
	var record promptAnswers
	if err := strictjson.DecodeBytes(data, &record); err != nil {
		return promptAnswers{}, generationPrompt{}, err
	}
	if record.Prompt == "" || len(record.Answers) == 0 || slices.Contains(record.Answers, "") {
		return promptAnswers{}, generationPrompt{}, errors.New("a generation record needs a prompt and nonempty answers")
	}
	tokens, err := encode(record.Prompt)
	if err != nil {
		return promptAnswers{}, generationPrompt{}, err
	}
	generation := generationPrompt{tokens: tokens}
	for _, answer := range record.Answers {
		answered, err := encode(record.Prompt + answer)
		if err != nil {
			return promptAnswers{}, generationPrompt{}, err
		}
		generation.length = max(generation.length, len(answered)-len(tokens))
	}
	return record, generation, nil
}

// generateAnswer continues a prompt greedily, the model's most likely token
// at a time, and answers with the continuation's text.
func generateAnswer(model *densecausal.Model, piece func(int) string, prompt generationPrompt) (HeldoutPrediction, error) {
	tokens := slices.Clone(prompt.tokens)
	var text strings.Builder
	for range prompt.length {
		logits, err := model.Logits(tokens)
		if err != nil {
			return HeldoutPrediction{}, err
		}
		last := logits[(len(tokens)-1)*model.Dims.Vocab:]
		next := slices.Index(last, slices.Max(last))
		tokens = append(tokens, next)
		text.WriteString(piece(next))
	}
	return HeldoutPrediction{Text: text.String()}, nil
}

// nextTokens answers a record with the model's most likely next token at
// each position but the last, teacher-forced on the record's own tokens.
func nextTokens(model *densecausal.Model, example HeldoutExample) (HeldoutPrediction, error) {
	tokens := dtype.ConvertSlice[int](example.Input)
	_, logits, err := model.Loss(tokens)
	if err != nil {
		return HeldoutPrediction{}, err
	}
	vocabulary := model.Dims.Vocab
	predicted := make([]float64, 0, len(tokens)-1)
	for position := range len(tokens) - 1 {
		row := logits[position*vocabulary : (position+1)*vocabulary]
		predicted = append(predicted, float64(slices.Index(row, slices.Max(row))))
	}
	return HeldoutPrediction{Values: predicted}, nil
}
