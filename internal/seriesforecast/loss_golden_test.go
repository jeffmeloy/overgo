package seriesforecast

import (
	"math"
	"testing"
)

// TestForecastLossMatchesReference pins the objective and its analytic
// gradient against the golden's torch loss_reference (computed at the
// reference's own forecast values, so the loss is isolated from forward
// parity).
func TestForecastLossMatchesReference(t *testing.T) {
	g := readGolden(t)
	ref := struct {
		Case   int
		Target []float32
		MSE    float64
		Quant  float64
		Total  float64
		Grad   [][]float32
	}{}
	var raw struct {
		LossReference struct {
			Case     int         `json:"case"`
			Target   []float32   `json:"target"`
			MSE      float64     `json:"mse"`
			Quantile float64     `json:"quantile"`
			Total    float64     `json:"total"`
			Grad     [][]float32 `json:"grad"`
		} `json:"loss_reference"`
	}
	readGrad(t, "timesfm_golden.json", &raw)
	ref.Case, ref.Target = raw.LossReference.Case, raw.LossReference.Target
	ref.MSE, ref.Quant, ref.Total = raw.LossReference.MSE, raw.LossReference.Quantile, raw.LossReference.Total
	ref.Grad = raw.LossReference.Grad

	forecast := flatten(g.Cases[ref.Case].Forecast)
	quantiles := 1 + len(g.Quantiles)
	mse, quant, total, grad, err := ForecastLoss(forecast, ref.Target, g.Quantiles, g.Horizon, quantiles)
	if err != nil {
		t.Fatal(err)
	}
	// Loss terms: pure f64 reductions; only f32 input rounding differs.
	const lossTol = 1e-5
	if math.Abs(mse-ref.MSE) > lossTol || math.Abs(quant-ref.Quant) > lossTol || math.Abs(total-ref.Total) > lossTol {
		t.Fatalf("loss %g/%g/%g, want %g/%g/%g", mse, quant, total, ref.MSE, ref.Quant, ref.Total)
	}
	wantGrad := flatten(ref.Grad)
	if len(grad) != len(wantGrad) {
		t.Fatalf("grad len %d, want %d", len(grad), len(wantGrad))
	}
	for i := range grad {
		if d := math.Abs(float64(grad[i]) - float64(wantGrad[i])); d > 1e-6 {
			t.Fatalf("grad[%d]: %g vs %g", i, grad[i], wantGrad[i])
		}
	}
}

// TestPerDimScaleBackwardMatchesGolden pins the isolated per-dim softplus
// query-scale VJP (the tenth grad golden; the attnlayer test covers it only
// in composition).
func TestPerDimScaleBackwardMatchesGolden(t *testing.T) {
	var g struct {
		HD      int       `json:"hd"`
		Factor  float64   `json:"factor"`
		Q       []float64 `json:"q"`
		Pds     []float64 `json:"pds"`
		Dout    []float64 `json:"dout"`
		GradQ   []float64 `json:"grad_q"`
		GradPds []float64 `json:"grad_pds"`
	}
	readGrad(t, "timesfm_perdimscale_grad_golden.json", &g)
	factor := math.Log2E / math.Sqrt(float64(g.HD))
	if math.Abs(factor-g.Factor) > 1e-9 {
		t.Fatalf("factor %g, want %g (log2(e)/sqrt(hd) is the derivation)", factor, g.Factor)
	}
	pds := f32(g.Pds)
	scale := compiledQueryScale(pds, g.HD)
	gradQ := make([]float32, g.HD)
	gradPds := make([]float32, g.HD)
	for dim := 0; dim < g.HD; dim++ {
		e := math.Exp(g.Pds[dim])
		gradPds[dim] = float32(g.Dout[dim] * g.Q[dim] * factor * e / (1 + e))
		gradQ[dim] = float32(g.Dout[dim] * float64(scale[dim]))
	}
	closeTo(t, "grad_q", gradQ, g.GradQ, wiredTol)
	closeTo(t, "grad_pds", gradPds, g.GradPds, wiredTol)
}
