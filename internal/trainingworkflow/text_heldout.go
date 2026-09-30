package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
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
	MaximumSequence  int
	Host             bool
	MaxProjectedWall time.Duration
}

// RunTextHeldout reads both of the objective's memberships as text files,
// trains the model on the training records' token sequences in turn, and
// judges it by held-out token accuracy: at each position of a held-out
// record, whether the model's most likely next token is the one that
// follows. The model's weights update in place, so the base and trained
// models answer in one process.
func RunTextHeldout(ctx context.Context, store *overgodb.Store, request TextHeldoutRequest) (HeldoutResult, error) {
	objective, err := LoadObjectiveAlias(ctx, store, request.Alias)
	if err != nil {
		return HeldoutResult{}, err
	}
	if objective.Kind != trainingprogram.ObjectiveTokenPrediction || objective.Metric != trainingprogram.MetricTokenAccuracy {
		return HeldoutResult{}, fmt.Errorf("training workflow: objective %q is %q judged by %q, not token prediction by token accuracy", request.Alias, objective.Kind, objective.Metric)
	}
	model, err := loadTrainableModel(request.ModelDirectory)
	if err != nil {
		return HeldoutResult{}, err
	}
	encode, err := loadTrainableEncoder(request.ModelDirectory)
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
	// Each record is one UTF-8 document; its example is its first sequence
	// tokens: the input is the sequence, the target every token after the
	// first.
	examples, _, err := ObjectiveExamples(ctx, store, objective, []byte("overgo/training/text-utf8/v1"), recipecontract.ModalityText,
		trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "utf-8"),
		func(example trainingdata.Example) (HeldoutExample, error) {
			if len(example.Values) != 1 {
				return HeldoutExample{}, errors.New("not one text document")
			}
			tokens, err := encode(string(example.Values[0].Data))
			if err != nil {
				return HeldoutExample{}, err
			}
			tokens = tokens[:min(len(tokens), sequence)]
			if len(tokens) < 2 {
				return HeldoutExample{}, fmt.Errorf("encodes to %d token(s); a next-token target needs two", len(tokens))
			}
			return HeldoutExample{Input: dtype.ConvertSlice[float64](tokens), Target: dtype.ConvertSlice[float64](tokens[1:])}, nil
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
			_, _, _, err := runTrainingState(model, batches, config.BaseLearningRate, config.Momentum, !request.Host, false, nil, nil, nil)
			return err
		},
		Predict: func(example HeldoutExample) (HeldoutPrediction, error) {
			return nextTokens(model, example)
		},
	})
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
