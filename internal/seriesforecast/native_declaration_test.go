package seriesforecast

import (
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/modeltest"
)

// nativeSeries is the golden's input, generated the same way on both sides:
// 10 + 0.05 t + 3 sin(2 pi t / 24) + 1.5 cos(2 pi t / 7) + ((t * 7919) % 101)
// / 101 - 0.5.
func nativeSeries(length int) []float32 {
	values := make([]float32, length)
	for t := range values {
		noise := float64((t*7919)%101)/101 - 0.5
		values[t] = float32(10 + 0.05*float64(t) + 3*math.Sin(2*math.Pi*float64(t)/24) + 1.5*math.Cos(2*math.Pi*float64(t)/7) + noise)
	}
	return values
}

// TestNativeDeclarationLoads holds the native forecaster to the model
// repository's own torch reference (testdata/timesfm3_golden.json): a
// context that fills whole patches, one that is front-padded and forecast
// over two output patches so overlapping patch forecasts are stitched, and a
// short one. Every quantile of every step must agree to float32 accumulation
// order, and Forecast returns the checkpoint's output patch with the median
// as its point column.
func TestNativeDeclarationLoads(t *testing.T) {
	t.Parallel()
	var golden struct {
		Cases []struct {
			Length    int       `json:"length"`
			Horizon   int       `json:"horizon"`
			Quantiles []float32 `json:"quantiles"`
		} `json:"cases"`
	}
	if err := jsonfile.Decode(filepath.Join("testdata", "timesfm3_golden.json"), &golden); err != nil {
		t.Fatal(err)
	}
	model, err := Load(modeltest.Directory(t, "timesfm-3.0-pytorch"))
	if err != nil {
		t.Fatal(err)
	}
	levels := len(model.Levels)
	for _, reference := range golden.Cases {
		out, err := model.forecastNative(nativeSeries(reference.Length), reference.Horizon)
		if err != nil {
			t.Fatal(err)
		}
		var worst, scale float64
		for step := range reference.Horizon {
			for level := range levels {
				got := float64(out[step*model.Dims.Quantiles+1+level])
				want := float64(reference.Quantiles[step*levels+level])
				worst = max(worst, math.Abs(got-want))
				scale = max(scale, math.Abs(want))
			}
		}
		t.Logf("context %d horizon %d: max abs diff %.3g of scale %.3g", reference.Length, reference.Horizon, worst, scale)
		if worst > 1e-5*scale {
			t.Errorf("context %d horizon %d differs from the reference: max abs diff %.3g", reference.Length, reference.Horizon, worst)
		}
	}
	forecast, err := model.Forecast(nativeSeries(golden.Cases[0].Length))
	if err != nil {
		t.Fatal(err)
	}
	if len(forecast) != model.Dims.Horizon*model.Dims.Quantiles {
		t.Fatalf("forecast holds %d values, want %d steps of %d columns", len(forecast), model.Dims.Horizon, model.Dims.Quantiles)
	}
	for step := range model.Dims.Horizon {
		row := forecast[step*model.Dims.Quantiles : (step+1)*model.Dims.Quantiles]
		if row[0] != row[1+levels/2] {
			t.Fatalf("step %d point %v is not the median %v", step, row[0], row[1+levels/2])
		}
	}
}
