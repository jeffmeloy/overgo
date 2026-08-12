package seriesforecast

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

var (
	forecastInputFixture  = []float32{3, 5}
	forecastOutputFixture = []float32{8, 13}
)

type forecastFunc func([]float32) ([]float32, error)

func (f forecastFunc) Forecast(series []float32) ([]float32, error) { return f(series) }

func TestRegisteredRuntimeExecutesIdentityBoundForecastProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "forecast-model", recipe.TaskForecast)
	if err := registerRuntime(fixture.Runtime, fixture.Model, forecastFunc(func(series []float32) ([]float32, error) {
		if !slices.Equal(series, forecastInputFixture) {
			t.Fatalf("series = %v", series)
		}
		return forecastOutputFixture, nil
	})); err != nil {
		t.Fatal(err)
	}
	got := modelrecipetest.MustExecuteScalar[[]float32](t, fixture, "forecast/runtime", forecastInputFixture)
	if !slices.Equal(got, forecastOutputFixture) {
		t.Fatalf("forecast = %v", got)
	}
}

func TestRegisteredRuntimeRejectsDifferentRecipeModel(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "forecast-other", recipe.TaskForecast)
	bound := testutil.ArtifactID(t, artifact.KindModel, "bound")
	if err := registerRuntime(fixture.Runtime, bound, forecastFunc(func(series []float32) ([]float32, error) {
		return series, nil
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.ExecuteScalar("forecast/mismatch", forecastInputFixture); err == nil {
		t.Fatal("mismatched model binding accepted")
	}
}
