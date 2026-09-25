package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
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

// runObjective fine-tunes a TabFM classification head under a registered
// table-prediction objective and publishes its held-out verdict: each
// record of the objective's registered dataset is one request file, the
// head trains only on the training membership's requests, and the base and
// trained heads are judged on the held-out requests' query rows.
func runObjective(storeRoot, alias, modelDir, task string, steps int) (err error) {
	ctx := context.Background()
	store, err := overgodb.Open(storeRoot)
	if err != nil {
		return err
	}
	defer store.Close()
	objectiveID, bound, err := store.ResolveAlias(ctx, alias)
	if err != nil {
		return err
	}
	if !bound {
		return fmt.Errorf("tabular-train-probe: objective alias %q is absent", alias)
	}
	objective, err := trainingprogram.LoadObjective(ctx, store, objectiveID)
	if err != nil {
		return err
	}
	if objective.Kind != trainingprogram.ObjectiveTablePrediction {
		return fmt.Errorf("tabular-train-probe: objective %q trains %q, not table prediction", alias, objective.Kind)
	}
	headDir := filepath.Join(modelDir, task)
	head, err := tabularicl.LoadHead(headDir)
	if err != nil {
		return err
	}
	if !head.Dims.IsClassifier {
		return fmt.Errorf("tabular-train-probe: table accuracy judges a classification head; %s is not one", headDir)
	}
	requests, examples, err := objectiveRequests(ctx, store, objective, task)
	if err != nil {
		return err
	}
	weights, err := filepath.Glob(filepath.Join(headDir, "*.safetensors"))
	if err != nil || len(weights) != 1 {
		return fmt.Errorf("tabular-train-probe: %s holds %d safetensors files, want one to record", headDir, len(weights))
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTablePrediction, Updates: steps,
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
	result, err := trainingworkflow.RunHeldout(ctx, store, trainingworkflow.HeldoutRun{
		// optimizer.NewStepper runs Muon on the CUDA device on Windows
		// (stepper_windows.go) and on the host elsewhere; the observation
		// records which.
		Alias: alias, ModelPath: weights[0], Examples: examples, Host: runtime.GOOS != "windows",
		Train: func(_ context.Context, shown []trainingworkflow.HeldoutExample) error {
			return trainTable(trainer, requests, shown, plan.Updates())
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			return predictClasses(head, requests[example.Record])
		},
	})
	if err != nil {
		return err
	}
	fmt.Printf("held-out verdict %s passed=%t observation=%s view=%s plan=%s baseline=%s candidate=%s\n",
		result.Verdict, result.Passed, result.Observation, result.View, result.Plan, result.Baseline, result.Candidate)
	return nil
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
			return nil, nil, fmt.Errorf("tabular-train-probe: %s membership: %w", membership.Partition, errors.Join(err, closeErr))
		}
		for _, example := range read {
			var request tabularicl.Request
			if len(example.Values) != 1 || json.Unmarshal(example.Values[0].Data, &request) != nil {
				return nil, nil, fmt.Errorf("tabular-train-probe: record %s is not one request document", example.ID)
			}
			if err := tabularicl.ValidateRequest(request); err != nil || request.Task != task {
				return nil, nil, fmt.Errorf("tabular-train-probe: record %s is not a valid %s request: %v", example.ID, task, err)
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
		return errors.New("tabular-train-probe: the training membership holds no requests")
	}
	for step := range updates {
		result, err := trainer.Step(requests[shown[step%len(shown)].Record])
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Loss) {
			return fmt.Errorf("tabular-train-probe: step %d loss is non-finite", step+1)
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
