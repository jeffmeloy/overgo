package seriesforecast

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/testutil"
)

var (
	forecastInputFixture  = []float32{3, 5}
	forecastOutputFixture = []float32{8, 13}
)

type forecastFunc func([]float32) ([]float32, error)

func (f forecastFunc) Forecast(series []float32) ([]float32, error) { return f(series) }

func TestRegisteredRuntimeExecutesIdentityBoundForecastProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "forecast-model", modelrecipe.ForecastDefinition)
	if err := registerRuntime(fixture.Runtime, fixture.Model, forecastFunc(func(series []float32) ([]float32, error) {
		if !slices.Equal(series, forecastInputFixture) {
			t.Fatalf("series = %v", series)
		}
		return forecastOutputFixture, nil
	})); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.ExecuteTensor("forecast/runtime", "series", forecastInputFixture)
	if err != nil {
		t.Fatal(err)
	}
	datum, one := result.Outputs["forecast"].Single()
	got, typed := datum.Value.([]float32)
	if !one || !typed || !slices.Equal(got, forecastOutputFixture) || !result.Commit.Valid() {
		t.Fatalf("forecast = (%v, %v, %v), commit=%v", got, one, typed, result.Commit)
	}
}

func TestRegisteredRuntimeRejectsDifferentRecipeModel(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "forecast-other", modelrecipe.ForecastDefinition)
	bound := testutil.ArtifactID(t, artifact.KindModel, "bound")
	if err := registerRuntime(fixture.Runtime, bound, forecastFunc(func(series []float32) ([]float32, error) {
		return series, nil
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.ExecuteTensor("forecast/mismatch", "series", forecastInputFixture); err == nil {
		t.Fatal("mismatched model binding accepted")
	}
}
