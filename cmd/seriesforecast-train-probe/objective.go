package main

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/seriesforecast"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// forecastSignature is the time-series pair a forecast objective trains.
var forecastSignature = recipecontract.ModalitySignature{
	Inputs:  []recipecontract.Modality{recipecontract.ModalityTimeSeries},
	Outputs: []recipecontract.Modality{recipecontract.ModalityTimeSeries},
}

// runObjective trains the model under a registered forecast objective and
// publishes its held-out verdict: both memberships of the objective's
// registered dataset are materialized through the training data path, the
// model trains only on the training membership, and the base and trained
// point forecasts are judged on the held-out membership.
func runObjective(storeRoot, alias, modelDir string, steps int, maxWall time.Duration) error {
	ctx := context.Background()
	store, err := overgodb.Open(storeRoot)
	if err != nil {
		return err
	}
	defer store.Close()
	objective, err := trainingworkflow.LoadObjectiveAlias(ctx, store, alias)
	if err != nil {
		return err
	}
	if objective.Kind != trainingprogram.ObjectiveForecast {
		return fmt.Errorf("seriesforecast-train-probe: objective %q trains %q, not forecasting", alias, objective.Kind)
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	modelDir = roots.ResolveModelPath(modelDir)
	model, err := seriesforecast.Load(modelDir)
	if err != nil {
		return err
	}
	examples, err := objectiveExamples(ctx, store, objective, model.Dims.PatchLen, model.Dims.Horizon)
	if err != nil {
		return err
	}
	weights, err := filepath.Glob(filepath.Join(modelDir, "*.safetensors"))
	if err != nil || len(weights) != 1 {
		return fmt.Errorf("seriesforecast-train-probe: %s holds %d safetensors files, want one to record", modelDir, len(weights))
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveForecast, Updates: steps, MaxProjectedWall: maxWall,
		Parameters: model.TrainableParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
	if err != nil {
		return err
	}
	trainer, err := seriesforecast.NewTrainer(model, plan.Optimizer())
	if err != nil {
		return err
	}
	defer trainer.Close()
	result, err := trainingworkflow.RunHeldout(ctx, store, trainingworkflow.HeldoutRun{
		// optimizer.NewStepper runs Muon on the CUDA device on Windows
		// (stepper_windows.go) and on the host elsewhere; the observation
		// records which.
		Alias: alias, ModelPath: weights[0], Examples: examples, Host: runtime.GOOS != "windows",
		Train: func(_ context.Context, shown []trainingworkflow.HeldoutExample) error {
			return trainForecast(trainer, shown, plan.Updates())
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			return pointForecast(model, example)
		},
	})
	if err != nil {
		return err
	}
	fmt.Printf("held-out verdict %s passed=%t observation=%s view=%s plan=%s baseline=%s candidate=%s\n",
		result.Verdict, result.Passed, result.Observation, result.View, result.Plan, result.Baseline, result.Candidate)
	return nil
}

// objectiveExamples materializes both of the objective's memberships with the
// light-curve processor and keys each example by its record identity.
func objectiveExamples(ctx context.Context, store *overgodb.Store, objective trainingprogram.ObjectiveDocument, patchLen, horizon int) (map[string]trainingworkflow.HeldoutExample, error) {
	training, heldout, err := trainingworkflow.ObjectiveMemberships(ctx, store, objective)
	if err != nil {
		return nil, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte(fmt.Sprintf("light-curve-train-v1/%d/%d", patchLen, horizon)))
	if err != nil {
		return nil, err
	}
	processor, err := seriesforecast.LightCurveTrainingProcessor(patchLen, horizon)
	if err != nil {
		return nil, err
	}
	examples := map[string]trainingworkflow.HeldoutExample{}
	for _, membership := range []dataset.Membership{training, heldout} {
		materialized, err := trainingdata.Materialize(ctx, store, trainingdata.Authority{
			Dataset: objective.Dataset, Split: membership.ID, Processors: []artifact.ID{processorID}, Signature: forecastSignature,
		}, []trainingdata.ProcessorBinding{{
			Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityTimeSeries}, Process: processor,
		}})
		if err != nil {
			return nil, err
		}
		batch, err := trainingworkflow.ReadAll(ctx, materialized)
		closeErr := materialized.Close()
		if err != nil || closeErr != nil {
			return nil, fmt.Errorf("seriesforecast-train-probe: %s membership: %w", membership.Partition, errors.Join(err, closeErr))
		}
		for _, example := range batch {
			input, target, err := seriesforecast.TrainingPair(example)
			if err != nil {
				return nil, fmt.Errorf("seriesforecast-train-probe: record %s: %w", example.ID, err)
			}
			examples[example.ID] = trainingworkflow.HeldoutExample{Input: dtype.ConvertSlice[float64](input), Target: dtype.ConvertSlice[float64](target)}
		}
	}
	return examples, nil
}

// trainForecast takes updates optimizer steps over the training examples in
// turn, refusing a step whose loss is not finite.
func trainForecast(trainer *seriesforecast.Trainer, shown []trainingworkflow.HeldoutExample, updates int) error {
	if len(shown) == 0 {
		return fmt.Errorf("seriesforecast-train-probe: the training membership holds no examples")
	}
	for step := range updates {
		example := shown[step%len(shown)]
		result, err := trainer.Step(dtype.ConvertSlice[float32](example.Input), dtype.ConvertSlice[float32](example.Target))
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Total) {
			return fmt.Errorf("seriesforecast-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: total=%.6f grad_l2=%.4g\n", step+1, updates, result.Total, result.GradientL2)
	}
	return nil
}

// pointForecast answers one example with the point channel of the model's
// forecast over as many steps as its target covers.
func pointForecast(model *seriesforecast.Model, example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
	forecast, err := model.Forecast(dtype.ConvertSlice[float32](example.Input))
	if err != nil {
		return trainingworkflow.HeldoutPrediction{}, err
	}
	if len(example.Target) > model.Dims.Horizon {
		return trainingworkflow.HeldoutPrediction{}, fmt.Errorf("seriesforecast-train-probe: target covers %d of horizon %d", len(example.Target), model.Dims.Horizon)
	}
	values := make([]float64, len(example.Target))
	for step := range values {
		values[step] = float64(forecast[step*model.Dims.Quantiles])
	}
	return trainingworkflow.HeldoutPrediction{Values: values}, nil
}
