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
	"overgo/internal/dataset"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/tabularicl"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// tableSignature is the table pair a table-prediction objective trains.
var tableSignature = recipecontract.ModalitySignature{
	Inputs:  []recipecontract.Modality{recipecontract.ModalityTable},
	Outputs: []recipecontract.Modality{recipecontract.ModalityTable},
}

// tableObjectiveTask is the head a table-prediction objective trains: its
// verdict is label accuracy, which only a classification head answers.
const tableObjectiveTask = "classification"

// tabularSteps is the tabular route's output: every decoder update's result.
type tabularSteps struct {
	Task       string                       `json:"task"`
	Parameters int                          `json:"parameters"`
	Steps      []tabularicl.TrainStepResult `json:"steps"`
}

// trainTabular is the tabular route: bounded decoder updates of a TabFM
// task head on one request, printed as JSON.
func trainTabular(flags *flag.FlagSet, args []string) (err error) {
	model := flags.String("model", "", "TabFM model directory")
	task := flags.String("task", "", "classification or regression")
	input := flags.String("input", "", "training request JSON")
	steps := clioptions.IntOverride(flags, "steps", "required decoder update steps")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *model == "" || *task == "" || *input == "" || *steps <= 0 {
		return fmt.Errorf("%s: model, task, input, and positive steps required", flags.Name())
	}
	var request tabularicl.Request
	if err := jsonfile.Decode(*input, &request); err != nil {
		return err
	}
	if request.Task != *task {
		return fmt.Errorf("%s: request task %q differs from %q", flags.Name(), request.Task, *task)
	}
	head, err := tabularicl.LoadHead(filepath.Join(*model, *task))
	if err != nil {
		return err
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTablePrediction, Updates: *steps,
		Parameters: head.TrainableParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
	if err != nil {
		return err
	}
	trainer, err := tabularicl.NewTrainer(head, plan.Optimizer())
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, trainer.Close()) }()
	result := tabularSteps{Task: *task, Parameters: trainer.ParameterCount(), Steps: make([]tabularicl.TrainStepResult, plan.Updates())}
	for index := range result.Steps {
		result.Steps[index], err = trainer.Step(request)
		if err != nil {
			return err
		}
	}
	return clioptions.WritePrettyJSON(os.Stdout, result)
}

// trainTableObjective fine-tunes a TabFM classification head under a
// registered table-prediction objective and publishes its held-out verdict:
// each record of the objective's registered dataset is one request file, the
// head trains only on the training membership's requests, and the base and
// trained heads are judged on the held-out requests' query rows.
func trainTableObjective(ctx context.Context, store *overgodb.Store, run objectiveRun, objective trainingprogram.ObjectiveDocument) (_ trainingworkflow.HeldoutResult, err error) {
	headDir := filepath.Join(run.Model, tableObjectiveTask)
	head, err := tabularicl.LoadHead(headDir)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	if !head.Dims.IsClassifier {
		return trainingworkflow.HeldoutResult{}, fmt.Errorf("table accuracy judges a classification head; %s is not one", headDir)
	}
	requests, examples, err := objectiveRequests(ctx, store, objective, tableObjectiveTask)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	weights, err := filepath.Glob(filepath.Join(headDir, "*.safetensors"))
	if err != nil || len(weights) != 1 {
		return trainingworkflow.HeldoutResult{}, fmt.Errorf("%s holds %d safetensors files, want one to record", headDir, len(weights))
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTablePrediction, Updates: run.Steps,
		Parameters: head.TrainableParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
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
		Alias: run.Alias, ModelPath: weights[0], Examples: examples, Host: runtime.GOOS != "windows",
		Train: func(_ context.Context, shown []trainingworkflow.HeldoutExample) error {
			return trainTable(trainer, requests, shown, plan.Updates())
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			return predictClasses(head, requests[example.Record])
		},
	})
}

// objectiveRequests reads both of the objective's memberships as request
// documents and keys each by its record: the example's target is the class
// of every query row.
func objectiveRequests(ctx context.Context, store *overgodb.Store, objective trainingprogram.ObjectiveDocument, task string) (map[string]tabularicl.Request, map[string]trainingworkflow.HeldoutExample, error) {
	training, heldout, err := trainingworkflow.ObjectiveMemberships(ctx, store, objective)
	if err != nil {
		return nil, nil, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/tabular-request/json/v1"))
	if err != nil {
		return nil, nil, err
	}
	requests := map[string]tabularicl.Request{}
	examples := map[string]trainingworkflow.HeldoutExample{}
	for _, membership := range []dataset.Membership{training, heldout} {
		materialized, err := trainingdata.Materialize(ctx, store, trainingdata.Authority{
			Dataset: objective.Dataset, Split: membership.ID, Processors: []artifact.ID{processorID}, Signature: tableSignature,
		}, []trainingdata.ProcessorBinding{{
			Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityTable},
			Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityTable, "json"),
		}})
		if err != nil {
			return nil, nil, err
		}
		read, err := trainingworkflow.ReadAll(ctx, materialized)
		closeErr := materialized.Close()
		if err != nil || closeErr != nil {
			return nil, nil, fmt.Errorf("%s membership: %w", membership.Partition, errors.Join(err, closeErr))
		}
		for _, example := range read {
			var request tabularicl.Request
			if len(example.Values) != 1 || json.Unmarshal(example.Values[0].Data, &request) != nil {
				return nil, nil, fmt.Errorf("record %s is not one request document", example.ID)
			}
			if err := tabularicl.ValidateRequest(request); err != nil || request.Task != task {
				return nil, nil, fmt.Errorf("record %s is not a valid %s request: %v", example.ID, task, err)
			}
			requests[example.ID] = request
			examples[example.ID] = trainingworkflow.HeldoutExample{Target: dtype.ConvertSlice[float64](request.Y[request.TrainRows:])}
		}
	}
	return requests, examples, nil
}

// trainTable takes updates optimizer steps over the training requests in
// turn, refusing a step whose loss is not finite.
func trainTable(trainer *tabularicl.Trainer, requests map[string]tabularicl.Request, shown []trainingworkflow.HeldoutExample, updates int) error {
	if len(shown) == 0 {
		return errors.New("the training membership holds no requests")
	}
	for step := range updates {
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

// predictClasses answers a request's query rows with each row's most
// likely class. The query rows' own labels are hidden from the head, which
// reads only the context rows' labels.
func predictClasses(head *tabularicl.Head, request tabularicl.Request) (trainingworkflow.HeldoutPrediction, error) {
	labels := slices.Clone(request.Y)
	clear(labels[request.TrainRows:])
	outputs, err := head.Predict(request.X, labels, request.Rows, request.Cols, request.TrainRows, request.CatCols)
	if err != nil {
		return trainingworkflow.HeldoutPrediction{}, err
	}
	width := head.Dims.OutDim
	classes := make([]float64, 0, request.Rows-request.TrainRows)
	for row := request.TrainRows; row < request.Rows; row++ {
		scores := outputs[row*width : (row+1)*width]
		classes = append(classes, float64(slices.Index(scores, slices.Max(scores))))
	}
	return trainingworkflow.HeldoutPrediction{Values: classes}, nil
}
