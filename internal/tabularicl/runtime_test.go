package tabularicl

import (
	"math"
	"reflect"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
)

const (
	tabularFixtureRows   = 2
	tabularFixtureCols   = 1
	tabularFixtureTrain  = 1
	tabularFixtureOutDim = 2
)

var (
	tabularRequestFixture = Request{
		Task: TaskClassification, X: []float32{2, 3}, Y: []float32{1, 0},
		Rows: tabularFixtureRows, Cols: tabularFixtureCols, TrainRows: tabularFixtureTrain,
	}
	tabularValuesFixture = []float32{5, 7, 11, 13}
	invalidValuesFixture = []float32{17}
)

type predictorFunc func(Request) ([]float32, int, error)

func (f predictorFunc) Predict(request Request) ([]float32, int, error) { return f(request) }

func TestLoadTaskRejectsUnknownHeadBeforeFilesystemAccess(t *testing.T) {
	if _, err := LoadTask(t.TempDir(), "unsupported-fixture-task"); err == nil {
		t.Fatal("unsupported task accepted")
	}
}

func TestValidateRequestRejectsMalformedInputs(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Request)
	}{
		{name: "fractional-class", mutate: func(request *Request) { request.Y[0] = 0.5 }},
		{name: "non-finite", mutate: func(request *Request) { request.X[0] = float32(math.NaN()) }},
		{name: "duplicate-categorical-column", mutate: func(request *Request) { request.CatCols = []int{0, 0} }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := tabularRequestFixture
			request.X = append([]float32(nil), request.X...)
			request.Y = append([]float32(nil), request.Y...)
			test.mutate(&request)
			if err := ValidateRequest(request); err == nil {
				t.Fatal("malformed request accepted")
			}
		})
	}
}

func TestRegisteredRuntimeEnforcesTabularOutputContract(t *testing.T) {
	tests := []struct {
		name      string
		values    []float32
		wantError bool
	}{
		{name: "valid", values: tabularValuesFixture},
		{name: "invalid-geometry", values: invalidValuesFixture, wantError: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := modelrecipetest.NewCapability(t, "tabular-"+test.name, modelrecipe.TabularDefinition)
			if err := registerRuntime(fixture.Runtime, fixture.Model, predictorFunc(
				func(request Request) ([]float32, int, error) {
					if !reflect.DeepEqual(request, tabularRequestFixture) {
						t.Fatalf("request = %+v", request)
					}
					return test.values, tabularFixtureOutDim, nil
				},
			)); err != nil {
				t.Fatal(err)
			}
			result, err := fixture.ExecuteTensor("tabular/runtime/"+test.name, "table", tabularRequestFixture)
			if test.wantError {
				if err == nil {
					t.Fatal("invalid prediction geometry accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			datum, ok := result.Outputs["predictions"].Single()
			prediction, typed := datum.Value.(Prediction)
			if !ok || !typed || prediction.Rows != tabularFixtureRows ||
				prediction.OutDim != tabularFixtureOutDim || !reflect.DeepEqual(prediction.Values, tabularValuesFixture) ||
				!result.Commit.Valid() {
				t.Fatalf("prediction = (%+v, %v, %v), commit=%v", prediction, ok, typed, result.Commit)
			}
		})
	}
}
