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

// trainTabular is the tabular route: fine-tuning a TabFM classification
// head's decoder under a registered table-prediction objective, whose
// held-out verdict and trained checkpoint it publishes.
func trainTabular(flags *flag.FlagSet, args []string) error {
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
	ctx := context.Background()
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

// trainTableObjective fine-tunes a TabFM classification head under a
// registered table-prediction objective and publishes its held-out verdict
// and the trained decoder's checkpoint at out: each record of the
// objective's registered dataset is one request file, the head trains only
// on the training membership's requests, and the base and trained heads are
// judged on the held-out requests' query rows.
func trainTableObjective(ctx context.Context, store *overgodb.Store, alias, model, out string, steps int) (_ trainingworkflow.HeldoutResult, err error) {
	objective, err := trainingworkflow.LoadObjectiveAlias(ctx, store, alias)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	if objective.Kind != trainingprogram.ObjectiveTablePrediction {
		return trainingworkflow.HeldoutResult{}, fmt.Errorf("objective %q trains %q, not table prediction", alias, objective.Kind)
	}
	headDir := filepath.Join(model, tableObjectiveTask)
	head, err := tabularicl.LoadHead(headDir)
	if err != nil {
		return trainingworkflow.HeldoutResult{}, err
	}
	if !head.Dims.IsClassifier {
		return trainingworkflow.HeldoutResult{}, fmt.Errorf("table accuracy judges a classification head; %s is not one", headDir)
	}
	requests, examples, training, err := objectiveRequests(ctx, store, objective, tableObjectiveTask)
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
		Alias: alias, ModelPath: weights, Examples: examples, Host: runtime.GOOS != "windows",
		Train: func(ctx context.Context, recipeID artifact.ID, shown []trainingworkflow.HeldoutExample) error {
			if err := trainTable(trainer, requests, shown, plan.Updates()); err != nil {
				return err
			}
			// Each update consumed one training request.
			return publishTrained(ctx, store, recipeID, out, trainer, trainingdata.StreamState{Identity: training, Position: uint64(plan.Updates())})
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			return predictClasses(head, requests[example.Record])
		},
	})
}

// objectiveRequests reads both of the objective's memberships as request
// documents and keys each by its record: the example's target is the class
// of every query row. It also returns the identity of the materialized
// training membership, the stream a checkpoint records its position in.
func objectiveRequests(ctx context.Context, store *overgodb.Store, objective trainingprogram.ObjectiveDocument, task string) (map[string]tabularicl.Request, map[string]trainingworkflow.HeldoutExample, artifact.ID, error) {
	training, heldout, err := trainingworkflow.ObjectiveMemberships(ctx, store, objective)
	if err != nil {
		return nil, nil, artifact.ID{}, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/tabular-request/json/v1"))
	if err != nil {
		return nil, nil, artifact.ID{}, err
	}
	requests := map[string]tabularicl.Request{}
	examples := map[string]trainingworkflow.HeldoutExample{}
	var stream artifact.ID
	for _, membership := range []dataset.Membership{training, heldout} {
		materialized, err := trainingdata.Materialize(ctx, store, trainingdata.Authority{
			Dataset: objective.Dataset, Split: membership.ID, Processors: []artifact.ID{processorID}, Signature: tableSignature,
		}, []trainingdata.ProcessorBinding{{
			Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityTable},
			Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityTable, "json"),
		}})
		if err != nil {
			return nil, nil, artifact.ID{}, err
		}
		if membership.ID == training.ID {
			stream = materialized.Identity()
		}
		read, err := trainingworkflow.ReadAll(ctx, materialized)
		closeErr := materialized.Close()
		if err != nil || closeErr != nil {
			return nil, nil, artifact.ID{}, fmt.Errorf("%s membership: %w", membership.Partition, errors.Join(err, closeErr))
		}
		for _, example := range read {
			var request tabularicl.Request
			if len(example.Values) != 1 || json.Unmarshal(example.Values[0].Data, &request) != nil {
				return nil, nil, artifact.ID{}, fmt.Errorf("record %s is not one request document", example.ID)
			}
			if err := tabularicl.ValidateRequest(request); err != nil || request.Task != task {
				return nil, nil, artifact.ID{}, fmt.Errorf("record %s is not a valid %s request: %v", example.ID, task, err)
			}
			requests[example.ID] = request
			examples[example.ID] = trainingworkflow.HeldoutExample{Target: dtype.ConvertSlice[float64](request.Y[request.TrainRows:])}
		}
	}
	return requests, examples, stream, nil
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
