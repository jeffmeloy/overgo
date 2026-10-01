package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/seriesforecast"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// runObjective trains the model under a registered forecast objective and
// publishes its held-out verdict and the trained checkpoint at out: both
// memberships of the objective's registered dataset are materialized through
// the training data path, the model trains only on the training membership,
// and the base and trained point forecasts are judged on the held-out
// membership.
func runObjective(ctx context.Context, storeRoot, alias, modelDir, out string, steps int, maxWall time.Duration) error {
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
		return fmt.Errorf("train -route seriesforecast:objective %q trains %q, not forecasting", alias, objective.Kind)
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
	data, err := objectiveExamples(ctx, store, objective, model.Dims.PatchLen, model.Dims.Horizon)
	if err != nil {
		return err
	}
	weights, err := trainingworkflow.IdentifyModelWeights(modelDir)
	if err != nil {
		return err
	}
	plan, err := probeSession(trainingprogram.ObjectiveForecast, steps, model.TrainableParameterCount(), maxWall)
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
		Alias: alias, ModelPath: weights, Data: data, Host: runtime.GOOS != "windows",
		Train: func(ctx context.Context, recipeID artifact.ID, shown []trainingworkflow.HeldoutExample) (trainingprogram.Checkpoint, error) {
			if err := trainForecast(ctx, trainer, shown, plan.Updates()); err != nil {
				return trainingprogram.Checkpoint{}, err
			}
			// Each update consumed one training record.
			return publishTrained(ctx, store, recipeID, out, trainer, trainingdata.StreamState{Identity: data.Training, Position: uint64(plan.Updates())})
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			return pointForecast(model, example)
		},
	})
	if err != nil {
		return err
	}
	return reportVerdict(os.Stdout, result)
}

// objectiveExamples materializes both of the objective's memberships with the
// light-curve processor and keys each example by its record identity.
func objectiveExamples(ctx context.Context, store *overgodb.Store, objective trainingprogram.ObjectiveDocument, patchLen, horizon int) (trainingworkflow.ObjectiveData, error) {
	processor, err := seriesforecast.LightCurveTrainingProcessor(patchLen, horizon)
	if err != nil {
		return trainingworkflow.ObjectiveData{}, err
	}
	return trainingworkflow.ObjectiveExamples(ctx, store, objective, fmt.Appendf(nil, "light-curve-train-v1/%d/%d", patchLen, horizon), recipecontract.ModalityTimeSeries, processor,
		func(example trainingdata.Example) (trainingworkflow.HeldoutExample, error) {
			input, target, err := seriesforecast.TrainingPair(example)
			return trainingworkflow.HeldoutExample{Input: dtype.ConvertSlice[float64](input), Target: dtype.ConvertSlice[float64](target)}, err
		})
}

// trainForecast takes updates optimizer steps over the training examples in
// turn until ctx ends, refusing a step whose loss is not finite.
func trainForecast(ctx context.Context, trainer *seriesforecast.Trainer, shown []trainingworkflow.HeldoutExample, updates int) error {
	if len(shown) == 0 {
		return fmt.Errorf("train -route seriesforecast:the training membership holds no examples")
	}
	for step := range updates {
		if err := ctx.Err(); err != nil {
			return err
		}
		example := shown[step%len(shown)]
		result, err := trainer.Step(dtype.ConvertSlice[float32](example.Input), dtype.ConvertSlice[float32](example.Target))
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Total) {
			return fmt.Errorf("train -route seriesforecast:step %d loss is non-finite", step+1)
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
		return trainingworkflow.HeldoutPrediction{}, fmt.Errorf("train -route seriesforecast:target covers %d of horizon %d", len(example.Target), model.Dims.Horizon)
	}
	values := make([]float64, len(example.Target))
	for step := range values {
		values[step] = float64(forecast[step*model.Dims.Quantiles])
	}
	return trainingworkflow.HeldoutPrediction{Values: values}, nil
}
