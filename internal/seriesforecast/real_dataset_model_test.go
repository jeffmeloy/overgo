//go:build modeltest

package seriesforecast

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

func TestRealSupernovaForecastStream(t *testing.T) {
	modelDir := modeltest.Directory(t, "timesfm-2.5-200m-transformers")
	datasetPath := supernovaDataset(t, "test")
	model, err := Load(modelDir)
	if err != nil {
		t.Fatalf("TimesFM artifact unavailable: %v", err)
	}
	record := leadingRecords(t, datasetPath, 1)[0]
	datasetID := identifyDataset(t, datasetPath)
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, []byte(datasetID.String()+"/heldout"))
	if err != nil {
		t.Fatal(err)
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, fmt.Appendf(nil, "light-curve-v1/%d/%d", model.Dims.PatchLen, model.Dims.Horizon))
	if err != nil {
		t.Fatal(err)
	}
	processor, err := LightCurveProcessor(model.Dims.PatchLen, model.Dims.Horizon)
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
	materialized, err := trainingdata.MaterializeDocuments(authority, processorID, []string{record}, trainingdata.ProcessorBinding{
		Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityTimeSeries}, Process: processor,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, nil)
	if err != nil {
		t.Fatal(err)
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := batcher.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	input, target, err := TrainingPair(batch.Examples[0])
	if err != nil {
		t.Fatal(err)
	}
	forecast, err := model.Forecast(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(forecast) != model.Dims.Horizon*model.Dims.Quantiles || len(target) == 0 || len(target) > model.Dims.Horizon {
		t.Fatalf("forecast/target geometry=%d/%d", len(forecast), len(target))
	}
	var modelError, persistenceError float64
	for index, want := range target {
		point := forecast[index*model.Dims.Quantiles]
		if math.IsNaN(float64(point)) || math.IsInf(float64(point), 0) {
			t.Fatalf("forecast[%d] is non-finite", index)
		}
		modelError += math.Abs(float64(point - want))
		persistenceError += math.Abs(float64(input[len(input)-1] - want))
	}
	modelError /= float64(len(target))
	persistenceError /= float64(len(target))
	if modelError >= persistenceError {
		t.Fatalf("TimesFM point MAE %.6f does not beat persistence %.6f", modelError, persistenceError)
	}
	t.Logf("real held-out light curve: context=%d target=%d point_MAE=%.6f persistence_MAE=%.6f first_forecast=%.6f", len(input), len(target), modelError, persistenceError, forecast[0])
}

func identifyDataset(t *testing.T, path string) artifact.ID {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("Supernova test dataset unavailable: %v", err)
	}
	defer file.Close()
	id, _, err := artifact.Identify(artifact.KindDataset, file)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func supernovaDataset(t *testing.T, split string) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id, found, err := store.ResolveAlias(t.Context(), "dataset.registered.supernova-timeseries")
	if err != nil || !found {
		t.Fatalf("Supernova dataset registration unavailable: found=%t, error=%v", found, err)
	}
	directory, err := artifact.AvailablePath(t.Context(), store, id, artifact.LocationDirectory)
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(directory, split+".jsonl")
}
