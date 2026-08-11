package seriesforecast

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

type forecastFunc func([]float32) ([]float32, error)

func (f forecastFunc) Forecast(series []float32) ([]float32, error) { return f(series) }

func TestRegisteredRuntimeExecutesIdentityBoundForecastProgram(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "forecast-model")
	publishRuntimeModel(t, store, modelID)
	definition, err := modelrecipe.ForecastDefinition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := workflowruntime.NewWithCatalog(store, modelrecipe.Catalog())
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{8, 13}
	if err := registerRuntime(runtime, modelID, forecastFunc(func(series []float32) ([]float32, error) {
		if !slices.Equal(series, []float32{3, 5}) {
			t.Fatalf("series = %v", series)
		}
		return want, nil
	})); err != nil {
		t.Fatal(err)
	}
	result, err := runtime.ExecuteProgram(context.Background(), "forecast/runtime", program, map[recipe.PortName]workflowruntime.Value{
		"series": {Kind: recipe.DataTensor, Items: []workflowruntime.Datum{{Value: []float32{3, 5}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, ok := result.Outputs["forecast"].Items[0].Value.([]float32)
	if !ok || !slices.Equal(got, want) || !result.Commit.Valid() {
		t.Fatalf("forecast = (%v, %v), commit=%v", got, ok, result.Commit)
	}
}

func TestRegisteredRuntimeRejectsDifferentRecipeModel(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	bound := testutil.ArtifactID(t, artifact.KindModel, "bound")
	other := testutil.ArtifactID(t, artifact.KindModel, "other")
	publishRuntimeModel(t, store, other)
	definition, _ := modelrecipe.ForecastDefinition(other)
	program, _ := modelrecipe.CompileCapability(definition)
	runtime, _ := workflowruntime.NewWithCatalog(store, modelrecipe.Catalog())
	if err := registerRuntime(runtime, bound, forecastFunc(func(series []float32) ([]float32, error) {
		return series, nil
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.ExecuteProgram(context.Background(), "forecast/mismatch", program, map[recipe.PortName]workflowruntime.Value{
		"series": {Kind: recipe.DataTensor, Items: []workflowruntime.Datum{{Value: []float32{1}}}},
	}); err == nil {
		t.Fatal("mismatched model binding accepted")
	}
}

func publishRuntimeModel(t *testing.T, store *repodb.Store, modelID artifact.ID) {
	t.Helper()
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "forecast/model/" + modelID.String(),
		Artifacts: []artifact.Descriptor{{ID: modelID}},
	}); err != nil {
		t.Fatal(err)
	}
}
