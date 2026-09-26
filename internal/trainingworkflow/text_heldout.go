package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
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
	weights, err := identifyModelWeights(request.ModelDirectory)
	if err != nil {
		return HeldoutResult{}, err
	}
	sequence := model.Dims.ContextLength
	if request.MaximumSequence > 0 {
		sequence = min(sequence, request.MaximumSequence)
	}
	examples, err := textObjectiveExamples(ctx, store, objective, encode, sequence)
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
		Train: func(_ context.Context, shown []HeldoutExample) error {
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

// textObjectiveExamples reads both memberships as UTF-8 text, one record per
// file, and turns each into its first sequence tokens: the input is the
// sequence, the target every token after the first.
func textObjectiveExamples(ctx context.Context, store *overgodb.Store, objective trainingprogram.ObjectiveDocument, encode func(string) ([]int, error), sequence int) (map[string]HeldoutExample, error) {
	training, heldout, err := ObjectiveMemberships(ctx, store, objective)
	if err != nil {
		return nil, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/text-utf8/v1"))
	if err != nil {
		return nil, err
	}
	examples := map[string]HeldoutExample{}
	for _, membership := range []dataset.Membership{training, heldout} {
		materialized, err := trainingdata.Materialize(ctx, store, trainingdata.Authority{
			Dataset: objective.Dataset, Split: membership.ID, Processors: []artifact.ID{processorID}, Signature: textSignature,
		}, []trainingdata.ProcessorBinding{{
			Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityText},
			Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "utf-8"),
		}})
		if err != nil {
			return nil, err
		}
		read, err := ReadAll(ctx, materialized)
		closeErr := materialized.Close()
		if err != nil || closeErr != nil {
			return nil, fmt.Errorf("training workflow: %s membership: %w", membership.Partition, errors.Join(err, closeErr))
		}
		for _, example := range read {
			if len(example.Values) != 1 {
				return nil, fmt.Errorf("training workflow: record %s is not one text document", example.ID)
			}
			tokens, err := encode(string(example.Values[0].Data))
			if err != nil {
				return nil, fmt.Errorf("training workflow: record %s: %w", example.ID, err)
			}
			tokens = tokens[:min(len(tokens), sequence)]
			if len(tokens) < 2 {
				return nil, fmt.Errorf("training workflow: record %s encodes to %d token(s); a next-token target needs two", example.ID, len(tokens))
			}
			examples[example.ID] = HeldoutExample{Input: dtype.ConvertSlice[float64](tokens), Target: dtype.ConvertSlice[float64](tokens[1:])}
		}
	}
	return examples, nil
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
