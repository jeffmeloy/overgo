package evaluation

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// lastValueForecaster forecasts that a series continues at its last value.
type lastValueForecaster struct{}

func (lastValueForecaster) Forecast(series []float32, steps int) ([]float32, error) {
	forecast := make([]float32, steps)
	for step := range forecast {
		forecast[step] = series[len(series)-1]
	}
	return forecast, nil
}

func (lastValueForecaster) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("a forecaster generates no text")
}

func (lastValueForecaster) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("a forecaster scores no continuation")
}

// textOnlyRuntime is a runtime that can neither forecast nor score.
type textOnlyRuntime struct{}

func (textOnlyRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", nil
}

func (textOnlyRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, nil
}

// TestSeriesForecastScoresTheMeanAbsoluteErrorOfEveryStep holds the report to
// the error of the forecast against the values that followed, over every step
// of every window, and refuses a runtime that cannot forecast.
func TestSeriesForecastScoresTheMeanAbsoluteErrorOfEveryStep(t *testing.T) {
	t.Parallel()
	suite := SeriesForecastSuite{Kind: seriesForecastKind, Schema: "fixture/v1", Source: "fixture", Cases: []SeriesForecastCase{
		{Name: "rising", Context: []float32{1, 2, 3}, Target: []float32{4, 5}},
		{Name: "flat", Context: []float32{7, 7}, Target: []float32{7, 9, 7}},
	}}
	compiled, err := compileSeriesForecast(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindSeriesForecast(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := evaluateSeriesForecast(t.Context(), store, lastValueForecaster{}, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	// The first window's forecast is 3, 3 against 4, 5: errors 1 and 2; the
	// second's is 7, 7, 7 against 7, 9, 7: errors 0, 2 and 0.
	if want := 5.0 / 5.0; report.Steps != 5 || math.Abs(report.MeanAbsoluteError-want) > 1e-12 ||
		report.Observations[0].MeanAbsoluteError != 1.5 || math.Abs(report.Observations[1].MeanAbsoluteError-2.0/3.0) > 1e-12 {
		t.Fatalf("report = %+v, want 5 steps and a mean absolute error of 1", report)
	}
	if _, err := evaluateSeriesForecast(t.Context(), store, textOnlyRuntime{}, compiled, plan); err == nil {
		t.Fatal("a runtime that cannot forecast was evaluated")
	}
}

// TestSeriesForecastRefusesWhatItCannotCompile holds the compiler to a case
// with nothing to show, nothing to forecast or a value that is no number.
func TestSeriesForecastRefusesWhatItCannotCompile(t *testing.T) {
	t.Parallel()
	valid := func() SeriesForecastSuite {
		return SeriesForecastSuite{Kind: seriesForecastKind, Schema: "fixture/v1", Source: "fixture", Cases: []SeriesForecastCase{
			{Name: "a", Context: []float32{1}, Target: []float32{2}}, {Name: "b", Context: []float32{1}, Target: []float32{2}},
		}}
	}
	if _, err := compileSeriesForecast(valid()); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*SeriesForecastSuite){
		"no context":           func(s *SeriesForecastSuite) { s.Cases[0].Context = nil },
		"no target":            func(s *SeriesForecastSuite) { s.Cases[0].Target = nil },
		"a value not a number": func(s *SeriesForecastSuite) { s.Cases[0].Target = []float32{float32(math.NaN())} },
		"a repeated case":      func(s *SeriesForecastSuite) { s.Cases[1].Name = "a" },
		"no cases":             func(s *SeriesForecastSuite) { s.Cases = nil },
	} {
		suite := valid()
		change(&suite)
		if _, err := compileSeriesForecast(suite); err == nil {
			t.Errorf("a suite with %s compiled", name)
		}
	}
}

// TestETTWindowsAreConsecutiveAndDoNotOverlap holds the builder to windows of
// the oil-temperature column: each shows its context and forecasts the readings
// after it, the next starts where it ends, and a window that would run past the
// file's end is left out.
func TestETTWindowsAreConsecutiveAndDoNotOverlap(t *testing.T) {
	t.Parallel()
	var table strings.Builder
	table.WriteString("date,HUFL,OT\n")
	for reading := range 11 {
		fmt.Fprintf(&table, "2016-07-01 %02d:00:00,%d,%d.5\n", reading, 100+reading, reading)
	}
	path := filepath.Join(t.TempDir(), "ETTh1.csv")
	if err := os.WriteFile(path, []byte(table.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	suite, err := BuildETTSeriesForecast(SeriesForecastSpec{CSV: path, Context: 3, Horizon: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(suite.Cases) != 2 || suite.Cases[0].Name != "etth1/oil-temperature/000" ||
		fmt.Sprint(suite.Cases[0].Context, suite.Cases[0].Target) != "[0.5 1.5 2.5] [3.5 4.5]" ||
		fmt.Sprint(suite.Cases[1].Context, suite.Cases[1].Target) != "[5.5 6.5 7.5] [8.5 9.5]" {
		t.Fatalf("suite = %+v, want two consecutive windows of the OT column", suite.Cases)
	}
	if _, err := compileSeriesForecast(suite); err != nil {
		t.Fatalf("the suite does not compile: %v", err)
	}
	for name, spec := range map[string]SeriesForecastSpec{
		"a window longer than the file": {CSV: path, Context: 10, Horizon: 5},
		"no horizon":                    {CSV: path, Context: 3},
		"no file":                       {CSV: path + ".absent", Context: 3, Horizon: 2},
	} {
		if _, err := BuildETTSeriesForecast(spec); err == nil {
			t.Errorf("a suite was built from %s", name)
		}
	}
}
