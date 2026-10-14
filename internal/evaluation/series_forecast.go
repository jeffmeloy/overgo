package evaluation

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	// seriesForecastKind names the suite whose cases give a model the start of
	// a series and score the point forecast it makes of the values after it.
	seriesForecastKind         = "series-forecast"
	seriesForecastMediaType    = "application/vnd.overgo.series-forecast-dataset+json"
	seriesForecastSchema       = "overgo/series-forecast-dataset/v1"
	seriesForecastSplitMedia   = "application/vnd.overgo.series-forecast-split+json"
	seriesForecastSplitSchema  = "overgo/series-forecast-split/v1"
	seriesForecastReportMedia  = "application/vnd.overgo.series-forecast-report+json"
	seriesForecastReportSchema = "overgo/series-forecast-report/v1"
)

var (
	seriesForecastDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: seriesForecastMediaType, Schema: seriesForecastSchema,
	}
	seriesForecastSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: seriesForecastSplitMedia, Schema: seriesForecastSplitSchema,
	}
	seriesForecastReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: seriesForecastReportMedia, Schema: seriesForecastReportSchema,
	}
)

// seriesForecaster is what a forecasting model offers a suite: the point
// forecast of the steps after a series.
type seriesForecaster interface {
	Forecast(series []float32, steps int) ([]float32, error)
}

// SeriesForecastCase is one window of a series: the values a model is shown
// and the values after them it forecasts.
type SeriesForecastCase struct {
	Name    string    `json:"name"`
	Context []float32 `json:"context"`
	Target  []float32 `json:"target"`
}

// SeriesForecastSuite is a set of windows from one series.
type SeriesForecastSuite struct {
	Kind   string               `json:"kind"`
	Schema string               `json:"schema"`
	Source string               `json:"source"`
	Cases  []SeriesForecastCase `json:"cases"`
}

type seriesForecastPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    SeriesForecastSuite
}

type seriesForecastObservation struct {
	Name              string  `json:"name"`
	MeanAbsoluteError float64 `json:"mean_absolute_error"`
}

type seriesForecastReport struct {
	ID                artifact.ID                 `json:"-"`
	Version           uint16                      `json:"version"`
	Plan              artifact.ID                 `json:"plan"`
	Dataset           artifact.ID                 `json:"dataset"`
	Observations      []seriesForecastObservation `json:"observations"`
	Steps             uint64                      `json:"steps"`
	MeanAbsoluteError float64                     `json:"mean_absolute_error"`
}

// compileSeriesForecast validates and identifies the suite.
func compileSeriesForecast(suite SeriesForecastSuite) (seriesForecastPlan, error) {
	if suite.Kind != seriesForecastKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return seriesForecastPlan{}, errors.New("evaluation: invalid series-forecast suite")
	}
	names := make(map[string]struct{}, len(suite.Cases))
	for _, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || len(testCase.Context) == 0 || len(testCase.Target) == 0 ||
			!finiteValues(testCase.Context) || !finiteValues(testCase.Target) {
			return seriesForecastPlan{}, errors.New("evaluation: invalid series-forecast case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return seriesForecastPlan{}, errors.New("evaluation: duplicate series-forecast case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, dataset, split, err := suiteIdentities(suite, suite.Cases)
	if err != nil {
		return seriesForecastPlan{}, err
	}
	return seriesForecastPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func finiteValues(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}

func bindSeriesForecast(compiled seriesForecastPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, seriesForecastKind, authorities)
}

// evaluateSeriesForecast asks the model for the point forecast of every
// window's target and publishes the mean absolute error of all the steps
// through the campaign ledger.
func evaluateSeriesForecast(
	ctx context.Context,
	repository artifact.Repository,
	runtime Runtime,
	compiled seriesForecastPlan,
	plan Plan,
) (seriesForecastReport, error) {
	forecaster, forecasts := runtime.(seriesForecaster)
	if ctx == nil || repository == nil || !forecasts || compiled.identity != plan.body.CaseProfile {
		return seriesForecastReport{}, errors.New("evaluation: series-forecast authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases, seriesForecastDatasetContract, seriesForecastSplitContract,
	)
	if err != nil {
		return seriesForecastReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return seriesForecastReport{}, err
	}
	report := seriesForecastReport{Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset}
	var total float64
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for _, testCase := range compiled.suite.Cases {
		forecast, err := forecaster.Forecast(testCase.Context, len(testCase.Target))
		if err != nil {
			return seriesForecastReport{}, fmt.Errorf("evaluation: series-forecast case %q: %w", testCase.Name, err)
		}
		if len(forecast) != len(testCase.Target) || !finiteValues(forecast) {
			return seriesForecastReport{}, fmt.Errorf("evaluation: series-forecast case %q: the forecast holds %d finite steps of the %d asked", testCase.Name, len(forecast), len(testCase.Target))
		}
		var errorSum float64
		for step, value := range testCase.Target {
			errorSum += math.Abs(float64(forecast[step]) - float64(value))
		}
		report.Observations = append(report.Observations, seriesForecastObservation{Name: testCase.Name, MeanAbsoluteError: errorSum / float64(len(testCase.Target))})
		total += errorSum
		report.Steps += uint64(len(testCase.Target))
		progress.advance()
	}
	report.MeanAbsoluteError = total / float64(report.Steps)
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return seriesForecastReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, seriesForecastReportContract, report,
	); err != nil {
		return seriesForecastReport{}, err
	}
	return report, nil
}

// seriesForecastMetrics reports forecast error: lower is better.
func seriesForecastMetrics(meanAbsoluteError float64) []runrecord.Metric {
	return []runrecord.Metric{{Name: "mean_absolute_error", Value: meanAbsoluteError, Direction: runrecord.DirectionMinimize}}
}
