//go:build modeltest

package diffusionimage

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

func TestRealCheckpointStructuredImageTraining(t *testing.T) {
	directory := artifactDir(t)
	paths, err := filepath.Glob(filepath.Join(directory, "epoch_*.png"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("UNAVAILABLE: structured image grids in %s", directory)
	}
	sources := make([]trainingdata.RawRecord, len(paths))
	for index, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		sources[index] = trainingdata.RawRecord{ID: filepath.Base(path), Group: filepath.Base(path), Data: data}
	}
	records, err := trainingdata.SelectStructuredImageCrops(sources, 32, 8)
	if err != nil {
		t.Fatal(err)
	}
	train := materializeImageBatch(t, records[:6], "train", 1)
	heldout := materializeImageBatch(t, records[6:], "heldout", 1)

	model, err := Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	parameters := 0
	for _, values := range model.raw {
		parameters += len(values)
	}
	baseLR := trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(parameters)
	trainer, err := NewTrainer(model, optimizer.Config{BaseLearningRate: baseLR, Momentum: 0.9, Schedule: optimizer.ScheduleConstant})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := trainer.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	const sigmaMin = 0.001
	trainLosses, trainErr := trainer.OTLoss(train, sigmaMin, 11)
	trainBefore := oneLoss(t, trainLosses, trainErr)
	heldoutLosses, heldoutErr := trainer.OTLoss(heldout, sigmaMin, 29)
	heldoutBefore := oneLoss(t, heldoutLosses, heldoutErr)
	projection := append([]float32(nil), model.raw["final_proj.weight"]...)
	stepLosses, stepErr := trainer.TrainOTBatch(train, sigmaMin, 11)
	stepLoss := oneLoss(t, stepLosses, stepErr)
	trainLosses, trainErr = trainer.OTLoss(train, sigmaMin, 11)
	trainAfter := oneLoss(t, trainLosses, trainErr)
	heldoutLosses, heldoutErr = trainer.OTLoss(heldout, sigmaMin, 29)
	heldoutAfter := oneLoss(t, heldoutLosses, heldoutErr)
	if math.Abs(stepLoss-trainBefore) > 1e-6 || !(trainAfter < trainBefore) {
		t.Fatalf("real train loss %.6f/%.6f -> %.6f", trainBefore, stepLoss, trainAfter)
	}
	if !finiteLoss(heldoutBefore) || !finiteLoss(heldoutAfter) || heldoutAfter > heldoutBefore*1.05 {
		t.Fatalf("held-out loss %.6f -> %.6f", heldoutBefore, heldoutAfter)
	}
	changed := 0
	for index, value := range model.raw["final_proj.weight"] {
		if value != projection[index] {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("real final projection did not update")
	}
	t.Logf("real SimpleDiffusion structured stream: records=6+2 params=%d lr=%.8g train=%.6f->%.6f heldout=%.6f->%.6f final_projection_changed=%d",
		parameters, baseLR, trainBefore, trainAfter, heldoutBefore, heldoutAfter, changed)
}

func materializeImageBatch(t *testing.T, records []trainingdata.RawRecord, split string, examples int) trainingdata.Batch {
	t.Helper()
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "simplediffusion-images")
	splitID := testutil.ArtifactID(t, artifact.KindDatasetShard, "simplediffusion-"+split)
	processorID := testutil.ArtifactID(t, artifact.KindProfile, "normalized-rgb")
	authority := trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID}, Shuffle: false,
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityImage}, Outputs: []recipecontract.Modality{recipecontract.ModalityImage}},
	}
	dataset, err := trainingdata.MaterializeRecords(authority, processorID, records, trainingdata.ProcessorBinding{
		Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityImage},
		Process: trainingdata.ImageProcessor(trainingdata.RoleTarget),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dataset.Close(); err != nil {
			t.Fatal(err)
		}
	})
	stream, err := trainingdata.NewStream(dataset, nil)
	if err != nil {
		t.Fatal(err)
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: examples, MicrobatchExamples: examples, DecodeWorkers: 1})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := batcher.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return batch
}

func oneLoss(t *testing.T, losses []float64, err error) float64 {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	if len(losses) != 1 || !finiteLoss(losses[0]) {
		t.Fatalf("losses = %v", losses)
	}
	return losses[0]
}

func finiteLoss(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
