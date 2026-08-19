//go:build modeltest

package seriesforecast

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

// TestTimesFMRealSupernovaTrainingSmoke: bounded observed training on the
// real TimesFM-2.5-200M checkpoint with a real supernova light curve from
// the TRAIN split — the training mirror of the inference validation path.
// Every step is observed (loss terms, learning rate, gradient norm, wall)
// and the first measured step arms a projection guard in place of a silent
// long run.
// trainingCandidateRecords bounds the scan for a trainable record.
const trainingCandidateRecords = 32

// inputSpread: max-min of the input window; zero means a constant window.
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

// leadingRecords returns up to limit leading JSONL records.
func leadingRecords(t *testing.T, path string, limit int) []string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("Supernova train dataset unavailable: %v", err)
	}
	defer file.Close()
	var records []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for len(records) < limit && scanner.Scan() {
		records = append(records, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(records) == 0 {
		t.Fatal("Supernova train dataset is empty")
	}
	return records
}

func TestTimesFMRealSupernovaTrainingSmoke(t *testing.T) {
	// The optimizer steps through optimizer.NewStepper — the shared CUDA Muon
	// on this host (host Newton-Schulz over a 203M-parameter pack runs serial
	// float64 at hours per step; the device stepper is the same parity-gated
	// pair the seq2seq training cells ran through). Descent is proven by
	// evaluating the same pair's loss before and after the observed steps.
	const steps = 3
	const maxWall = 25 * time.Minute
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	modelDir := filepath.Join(roots.Models, "timesfm-2.5-200m-transformers")
	datasetPath := filepath.Join(roots.Datasets, "supernova-timeseries", "train.jsonl")
	model, err := Load(modelDir)
	if err != nil {
		t.Fatalf("TimesFM artifact unavailable: %v", err)
	}
	records := leadingRecords(t, datasetPath, trainingCandidateRecords)
	datasetID := identifyDataset(t, datasetPath)
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, []byte(datasetID.String()+"/train"))
	if err != nil {
		t.Fatal(err)
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte(fmt.Sprintf("light-curve-train-v1/%d/%d", model.Dims.PatchLen, model.Dims.Horizon)))
	if err != nil {
		t.Fatal(err)
	}
	processor, err := LightCurveTrainingProcessor(model.Dims.PatchLen, model.Dims.Horizon)
	if err != nil {
		t.Fatal(err)
	}
	authority := trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityTimeSeries},
			Outputs: []recipecontract.Modality{recipecontract.ModalityTimeSeries},
		},
	}

	// The training pairing refuses curves too short for a two-patch context,
	// and a pre-explosion window is legitimately untrainable: constant input
	// makes the running sigma zero and the denormalized forecast carries no
	// weight dependence — the gradient is exactly zero. Records materialize
	// one at a time so each refusal is a logged skip, and the first record
	// with a varying multi-patch context trains.
	var input, target []float32
	ordinal := -1
	for index := 0; index < len(records) && ordinal < 0; index++ {
		candidateInput, candidateTarget, err := func() ([]float32, []float32, error) {
			materialized, err := trainingdata.MaterializeDocuments(authority, processorID, records[index:index+1], trainingdata.ProcessorBinding{
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
			return TrainingPair(batch.Examples[0])
		}()
		if err != nil {
			t.Logf("record %d skipped: %v", index, err)
			continue
		}
		if inputSpread(candidateInput) == 0 {
			t.Logf("record %d skipped: constant %d-point input window (zero gradient by construction)", index, len(candidateInput))
			continue
		}
		input, target, ordinal = candidateInput, candidateTarget, index
	}
	if ordinal < 0 {
		t.Fatalf("no trainable record in the first %d of the train split", len(records))
	}
	t.Logf("training record ordinal=%d input_spread=%.6g", ordinal, inputSpread(input))

	trainer, err := NewTrainer(model, optimizer.Config{Momentum: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	t.Logf("trainable parameters=%d dataset=%s split=%s context=%d horizon=%d",
		trainer.ParameterCount(), datasetID, splitID, len(input), len(target))

	start := time.Now()
	var first TrainStepResult
	for step := 0; step < steps; step++ {
		stepStart := time.Now()
		result, err := trainer.Step(input, target)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(result.Total) || math.IsInf(result.Total, 0) {
			t.Fatalf("step %d loss is non-finite: %+v", step+1, result)
		}
		t.Logf("step %d/%d: total=%.6f mse=%.6f quantile=%.6f lr=%.4g grad_l2=%.4g wall=%s",
			step+1, steps, result.Total, result.MSE, result.Quantile,
			result.LearningRate, result.GradientL2, time.Since(stepStart).Round(time.Millisecond))
		if step == 0 {
			first = result
			if result.GradientL2 <= 0 {
				t.Fatalf("first step carried no gradient: %+v", result)
			}
			if projected := time.Duration(steps) * time.Since(start); projected > maxWall {
				t.Fatalf("first step projects the run to %s, past the %s bound", projected.Round(time.Second), maxWall)
			}
		}
	}
	after, err := trainer.Loss(input, target)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(after) || math.IsInf(after, 0) || !(after < first.Total) {
		t.Fatalf("real training loss did not descend: before=%.6f after=%.6f", first.Total, after)
	}
	t.Logf("real supernova training smoke: steps=%d span_tokens=%d loss %.6f -> %.6f total_wall=%s",
		steps, steps*len(input), first.Total, after, time.Since(start).Round(time.Millisecond))
}
