package tabularicl

import (
	"math"
	"testing"
)

const (
	tabularFixtureRows  = 2
	tabularFixtureCols  = 1
	tabularFixtureTrain = 1
)

var tabularRequestFixture = Request{
	Task: TaskClassification, X: []float32{2, 3}, Y: []float32{1, 0},
	Rows: tabularFixtureRows, Cols: tabularFixtureCols, TrainRows: tabularFixtureTrain,
}

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
