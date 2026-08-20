// seriesforecast-train-probe: bounded forecast training evidence.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/seriesforecast"
	"overgo/internal/trainingdata"
)

func main() {
	model := flag.String("model", "", "forecast model directory (safetensors + config.json)")
	dataset := flag.String("dataset", "", "light-curve JSONL shard")
	steps := flag.Int("steps", 3, "observed Muon steps")
	records := flag.Int("records", 32, "leading records scanned for a trainable pair")
	maxWall := flag.Duration("max-wall", 25*time.Minute, "abort when the first measured step projects the run past this bound")
	flag.Parse()
	if err := run(*model, *dataset, *steps, *records, *maxWall); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(modelDir, datasetPath string, steps, recordLimit int, maxWall time.Duration) error {
	if modelDir == "" || datasetPath == "" || steps <= 0 || recordLimit <= 0 {
		return fmt.Errorf("seriesforecast-train-probe: -model and -dataset are required; -steps and -records must be positive")
	}
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
	trainer, err := seriesforecast.NewTrainer(model, optimizer.Config{Momentum: optimizer.DeriveMomentum()})
	if err != nil {
		return err
	}
	defer trainer.Close()
	fmt.Printf("trainable parameters=%d dataset=%s record=%d context=%d covered_horizon=%d/%d\n",
		trainer.ParameterCount(), datasetID, ordinal, len(input), len(target), model.Dims.Horizon)

	start := time.Now()
	var before float64
	for step := 0; step < steps; step++ {
		stepStart := time.Now()
		result, err := trainer.Step(input, target)
		if err != nil {
			return err
		}
		if math.IsNaN(result.Total) || math.IsInf(result.Total, 0) {
			return fmt.Errorf("seriesforecast-train-probe: step %d loss is non-finite", step+1)
		}
		fmt.Printf("step %d/%d: total=%.6f mse=%.6f quantile=%.6f lr=%.4g grad_l2=%.4g wall=%s\n",
			step+1, steps, result.Total, result.MSE, result.Quantile,
			result.LearningRate, result.GradientL2, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			before = result.Total
			if result.GradientL2 <= 0 {
				return fmt.Errorf("seriesforecast-train-probe: first step carried no gradient")
			}
			if projected := time.Duration(steps) * time.Since(start); projected > maxWall {
				return fmt.Errorf("seriesforecast-train-probe: first step projects the run to %s, past the %s bound", projected.Round(time.Second), maxWall)
			}
		}
	}
	after, err := trainer.Loss(input, target)
	if err != nil {
		return err
	}
	if math.IsNaN(after) || math.IsInf(after, 0) || !(after < before) {
		return fmt.Errorf("seriesforecast-train-probe: loss did not descend: before=%.6f after=%.6f", before, after)
	}
	fmt.Printf("descent: steps=%d span_tokens=%d loss %.6f -> %.6f total_wall=%s\n",
		steps, steps*len(input), before, after, time.Since(start).Round(time.Millisecond))
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
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte(fmt.Sprintf("light-curve-train-v1/%d/%d", model.Dims.PatchLen, model.Dims.Horizon)))
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
	return nil, nil, 0, fmt.Errorf("seriesforecast-train-probe: no trainable record in the first %d of the shard", len(records))
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
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for len(records) < limit && scanner.Scan() {
		records = append(records, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("seriesforecast-train-probe: dataset %s is empty", path)
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
