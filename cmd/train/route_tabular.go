//overgo:runtime-inputs caller

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/tabularicl"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// tableObjectiveTasks names the head a table-prediction objective trains by
// the metric its verdict declares: label accuracy only a classification head
// answers, squared error only a regression head.
var tableObjectiveTasks = map[trainingprogram.EvaluationMetric]string{
	trainingprogram.MetricTableAccuracy: tabularicl.TaskClassification,
	trainingprogram.MetricTableMSE:      tabularicl.TaskRegression,
}

// trainTabular is the tabular route: fine-tuning the decoder of the TabFM
// head its registered table-prediction objective's metric names, and
// publishing the objective's held-out verdict and the trained checkpoint.
func trainTabular(ctx context.Context, flags *flag.FlagSet, args []string) error {
	model := flags.String("model", "", "TabFM model directory")
	objectiveAlias := flags.String("objective", "", "registered table-prediction objective alias: train on its training membership, one request JSON per dataset file, and judge on its held-out one")
	steps := clioptions.IntOverride(flags, "steps", "required decoder update steps")
	storePath := flags.String("store", "overgodb-store", "OvergoDB holding the objective, and receiving the session observation, verdict and checkpoint record")
	out := flags.String("out", "", "directory the trained decoder's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *model == "" || *objectiveAlias == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("%s: model, objective, out, and positive steps required", flags.Name())
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	result, err := trainTableObjective(ctx, store, *objectiveAlias, *model, *out, *steps)
	if err != nil {
		return err
	}
	return reportVerdict(os.Stdout, result)
}

// trainTableObjective fine-tunes the TabFM head a registered
// table-prediction objective's metric names and publishes its held-out
// verdict and the trained decoder's checkpoint at out: each record of the
// objective's registered dataset is one request file, the head trains only
// on the training membership's requests, and the base and trained heads are
// judged on the held-out requests' query rows.
func trainTableObjective(ctx context.Context, store *overgodb.Store, alias, model, out string, steps int) (_ trainingworkflow.HeldoutResult, err error) {
	objective, err := trainingworkflow.LoadObjectiveAlias(ctx, store, alias)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	task, found := tableObjectiveTasks[objective.Metric]
	if objective.Kind != trainingprogram.ObjectiveTablePrediction || !found {
		return trainingworkflow.HeldoutResult{}, fmt.Errorf("objective %q trains %q judged by %q, which no TabFM head answers", alias, objective.Kind, objective.Metric)
	}
	headDir := filepath.Join(model, task)
	head, err := tabularicl.LoadHead(headDir)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	if head.Dims.IsClassifier != (task == tabularicl.TaskClassification) {
		return trainingworkflow.HeldoutResult{}, fmt.Errorf("table metric %q judges a %s head; %s is not one", objective.Metric, task, headDir)
	}
	requests, data, err := objectiveRequests(ctx, store, objective, task)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	weights, err := trainingworkflow.IdentifyModelWeights(headDir)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	plan, err := probeSession(trainingprogram.ObjectiveTablePrediction, steps, head.TrainableParameterCount(), 0)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	trainer, err := tabularicl.NewTrainer(head, plan.Optimizer())
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	defer func() { err = errors.Join(err, trainer.Close()) }()
	return trainingworkflow.RunHeldout(ctx, store, trainingworkflow.HeldoutRun{
		// optimizer.NewStepper runs Muon on the CUDA device on Windows
		// (stepper_windows.go) and on the host elsewhere; the observation
		// records which.
		Alias: alias, ModelPath: weights, Data: data, Host: runtime.GOOS != "windows",
		Train: func(ctx context.Context, recipeID artifact.ID, shown []trainingworkflow.HeldoutExample) (trainingprogram.Checkpoint, error) {
			if err := trainTable(ctx, trainer, requests, shown, plan.Updates()); err != nil {
				return trainingprogram.Checkpoint{}, err
			}
			// Each update consumed one training request.
			return publishTrained(ctx, store, recipeID, out, trainer, trainingdata.StreamState{Identity: data.Training, Position: uint64(plan.Updates())})
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			return predictQueryRows(head, requests[example.Record])
		},
	})
}

// objectiveRequests reads both of the objective's memberships as request
// documents and keys each by its record: the example's target is every query
// row's class or value.
func objectiveRequests(ctx context.Context, store *overgodb.Store, objective trainingprogram.ObjectiveDocument, task string) (map[string]tabularicl.Request, trainingworkflow.ObjectiveData, error) {
	requests := map[string]tabularicl.Request{}
	data, err := trainingworkflow.ObjectiveExamples(ctx, store, objective, []byte("overgo/tabular-request/json/v1"), recipecontract.ModalityTable,
		trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityTable, "json"),
		func(example trainingdata.Example) (trainingworkflow.HeldoutExample, error) {
			var request tabularicl.Request
			if len(example.Values) != 1 || json.Unmarshal(example.Values[0].Data, &request) != nil {
				return trainingworkflow.HeldoutExample{}, errors.New("not one request document")
			}
			if err := tabularicl.ValidateRequest(request); err != nil || request.Task != task {
				return trainingworkflow.HeldoutExample{}, fmt.Errorf("not a valid %s request: %v", task, err)
			}
			requests[example.ID] = request
			return trainingworkflow.HeldoutExample{Target: dtype.ConvertSlice[float64](request.Y[request.TrainRows:])}, nil
		})
	return requests, data, err
}

// trainTable takes updates optimizer steps over the training requests in
// turn until ctx ends, refusing a step whose loss is not finite.
func trainTable(ctx context.Context, trainer *tabularicl.Trainer, requests map[string]tabularicl.Request, shown []trainingworkflow.HeldoutExample, updates int) error {
	if len(shown) == 0 {
		return errors.New("the training membership holds no requests")
	}
	for step := range updates {
		if err := ctx.Err(); err != nil {
			return err
		}
		result, err := trainer.Step(requests[shown[step%len(shown)].Record])
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Loss) {
			return fmt.Errorf("step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: loss=%.6f\n", step+1, updates, result.Loss)
	}
	return nil
}

// predictQueryRows answers a request's query rows: a classification head
// with each row's most likely class, a regression head with each row's one
// emitted value. The query rows' own targets are hidden from the head, which
// reads only the context rows' targets.
func predictQueryRows(head *tabularicl.Head, request tabularicl.Request) (trainingworkflow.HeldoutPrediction, error) {
	targets := slices.Clone(request.Y)
	clear(targets[request.TrainRows:])
	outputs, err := head.Predict(request.X, targets, request.Rows, request.Cols, request.TrainRows, request.CatCols)
	if err != nil {
		return trainingworkflow.HeldoutPrediction{}, err
	}
	width := head.Dims.OutDim
	values := make([]float64, 0, request.Rows-request.TrainRows)
	for row := request.TrainRows; row < request.Rows; row++ {
		scores := outputs[row*width : (row+1)*width]
		value := float64(scores[0])
		if head.Dims.IsClassifier {
			value = float64(slices.Index(scores, slices.Max(scores)))
		}
		values = append(values, value)
	}
	return trainingworkflow.HeldoutPrediction{Values: values}, nil
}
