package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/recipecontract"
	"overgo/internal/seriesforecast"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// A light-curve shard line's scanner buffer: it starts at a megabyte and
// grows to at most sixteen, the longest record a shard holds.
const (
	shardLineBuffer = 1 << 20
	shardLineLimit  = 1 << 24
)

// trainSeriesForecast is the seriesforecast route: bounded forecast training
// evidence on a light-curve shard, or, with -objective, training under a
// registered forecast objective whose held-out verdict it publishes.
func trainSeriesForecast(flags *flag.FlagSet, args []string) error {
	modelDir := flags.String("model", "", "forecast model directory (safetensors + config.json)")
	datasetPath := flags.String("dataset", "", "light-curve JSONL shard")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	recordLimit := clioptions.IntOverride(flags, "records", "leading record scan bound; omitted scans the shard")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storeRoot := flags.String("store", "", "OvergoDB root; with -objective, train under that objective and publish its held-out verdict")
	objectiveAlias := flags.String("objective", "", "registered forecast objective alias (requires -store): train on its training membership, one light-curve JSON per dataset file, and judge on its held-out one; -dataset and -records are unused")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *objectiveAlias != "" {
		if *storeRoot == "" || *modelDir == "" || *steps <= 0 {
			return fmt.Errorf("%s: -objective requires -store, a model, and positive steps", flags.Name())
		}
		return runObjective(*storeRoot, *objectiveAlias, *modelDir, *steps, *maxWall)
	}
	if *modelDir == "" || *datasetPath == "" || *steps <= 0 || *recordLimit < 0 {
		return fmt.Errorf("%s: model, dataset, positive steps, and nonnegative record bound required", flags.Name())
	}
	return forecastShard(*modelDir, *datasetPath, *steps, *recordLimit, *maxWall)
}

// forecastShard trains the forecast model on the first trainable pair of a
// light-curve shard's leading records and proves the loss descends.
func forecastShard(modelDir, datasetPath string, steps, recordLimit int, maxWall time.Duration) error {
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	model, err := seriesforecast.Load(roots.ResolveModelPath(modelDir))
	if err != nil {
		return err
	}
	records, err := leadingRecords(datasetPath, recordLimit)
	if err != nil {
		return err
	}
	datasetID, err := identifyDataset(datasetPath)
	if err != nil {
		return err
	}
	input, target, ordinal, err := firstTrainablePair(model, datasetID, records)
	if err != nil {
		return err
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
	fmt.Printf("trainable parameters=%d dataset=%s record=%d context=%d covered_horizon=%d/%d\n",
		trainer.ParameterCount(), datasetID, ordinal, len(input), len(target), model.Dims.Horizon)

	start := time.Now()
	var before float64
	for step := 0; step < plan.Updates(); step++ {
		stepStart := time.Now()
		result, err := trainer.Step(input, target)
		if err != nil {
			return err
		}
		if !checked.Finite64(result.Total) {
			return fmt.Errorf("train -route seriesforecast:step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: total=%.6f mse=%.6f quantile=%.6f lr=%.4g grad_l2=%.4g wall=%s\n",
			step+1, plan.Updates(), result.Total, result.MSE, result.Quantile,
			result.LearningRate, result.GradientL2, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			before = result.Total
			if result.GradientL2 <= 0 {
				return fmt.Errorf("train -route seriesforecast:first step carried no gradient")
			}
			if err := plan.AdmitStepWall(time.Since(start)); err != nil {
				return fmt.Errorf("train -route seriesforecast:%w", err)
			}
		}
	}
	after, err := trainer.Loss(input, target)
	if err != nil {
		return err
	}
	if !checked.Finite64(after) || !(after < before) {
		return fmt.Errorf("train -route seriesforecast:loss did not descend: before=%.6f after=%.6f", before, after)
	}
	fmt.Printf("descent: steps=%d span_tokens=%d loss %.6f -> %.6f total_wall=%s\n",
		plan.Updates(), plan.Updates()*len(input), before, after, time.Since(start).Round(time.Millisecond))
	return nil
}

// firstTrainablePair scans the leading records for the first varying
// multi-patch context; refusals (short curves, constant windows) are logged
// skips, never silent.
func firstTrainablePair(model *seriesforecast.Model, datasetID artifact.ID, records []string) ([]float32, []float32, int, error) {
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, []byte(datasetID.String()+"/train"))
	if err != nil {
		return nil, nil, 0, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, fmt.Appendf(nil, "light-curve-train-v1/%d/%d", model.Dims.PatchLen, model.Dims.Horizon))
	if err != nil {
		return nil, nil, 0, err
	}
	processor, err := seriesforecast.LightCurveTrainingProcessor(model.Dims.PatchLen, model.Dims.Horizon)
	if err != nil {
		return nil, nil, 0, err
	}
	authority := trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityTimeSeries},
			Outputs: []recipecontract.Modality{recipecontract.ModalityTimeSeries},
		},
	}
	for index, record := range records {
		input, target, err := materializePair(authority, processorID, processor, record)
		if err != nil {
			fmt.Printf("record %d skipped: %v\n", index, err)
			continue
		}
		if inputSpread(input) == 0 {
			fmt.Printf("record %d skipped: constant %d-point input window (zero gradient by construction)\n", index, len(input))
			continue
		}
		fmt.Printf("training record ordinal=%d input_spread=%.6g\n", index, inputSpread(input))
		return input, target, index, nil
	}
	return nil, nil, 0, fmt.Errorf("train -route seriesforecast:no trainable record in the first %d of the shard", len(records))
}

func materializePair(authority trainingdata.Authority, processorID artifact.ID, processor trainingdata.Processor, record string) ([]float32, []float32, error) {
	materialized, err := trainingdata.MaterializeDocuments(authority, processorID, []string{record}, trainingdata.ProcessorBinding{
		Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityTimeSeries}, Process: processor,
	})
	if err != nil {
		return nil, nil, err
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, nil)
	if err != nil {
		return nil, nil, err
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1})
	if err != nil {
		return nil, nil, err
	}
	batch, err := batcher.Next(context.Background())
	if err != nil {
		return nil, nil, err
	}
	return seriesforecast.TrainingPair(batch.Examples[0])
}

func inputSpread(values []float32) float64 {
	if len(values) == 0 {
		return 0
	}
	lowest, highest := values[0], values[0]
	for _, v := range values {
		lowest, highest = min(lowest, v), max(highest, v)
	}
	return float64(highest - lowest)
}

func leadingRecords(path string, limit int) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var records []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, shardLineBuffer), shardLineLimit)
	for (limit == 0 || len(records) < limit) && scanner.Scan() {
		records = append(records, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("train -route seriesforecast:dataset %s is empty", path)
	}
	return records, nil
}

func identifyDataset(path string) (artifact.ID, error) {
	file, err := os.Open(path)
	if err != nil {
		return artifact.ID{}, err
	}
	defer file.Close()
	id, _, err := artifact.Identify(artifact.KindDataset, file)
	return id, err
}
