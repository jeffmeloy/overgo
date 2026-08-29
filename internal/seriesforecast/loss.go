// Training loss for the forecast capability: MSE on the point column plus
// mean pinball (quantile) loss over the level columns — the reference's
// objective. The gradient is analytic; the loss-reference golden pins both.
package seriesforecast

import "fmt"

// ForecastLoss: forecast is [horizon*quantiles] flat t-major (column 0 the
// point, columns 1.. the quantile levels); target is [horizon]. Returns the
// two loss terms, their total, and dTotal/dForecast.
func ForecastLoss(forecast, target []float32, levels []float64, horizon, quantiles int) (mse, quantile, total float64, grad []float32, err error) {
	if len(forecast) != horizon*quantiles || len(target) != horizon || len(levels) != quantiles-1 {
		return 0, 0, 0, nil, fmt.Errorf("seriesforecast: loss forecast=%d target=%d levels=%d horizon=%d quantiles=%d",
			len(forecast), len(target), len(levels), horizon, quantiles)
	}
	grad = make([]float32, len(forecast))
	invH := 1.0 / float64(horizon)
	invHQ := 1.0 / float64(horizon*(quantiles-1))
	for t := range horizon {
		y := float64(target[t])
		point := float64(forecast[t*quantiles])
		diff := point - y
		mse += diff * diff * invH
		grad[t*quantiles] = float32(2 * diff * invH)
		for j := 1; j < quantiles; j++ {
			tau := levels[j-1]
			pred := float64(forecast[t*quantiles+j])
			e := y - pred
			if e >= 0 {
				quantile += tau * e * invHQ
				grad[t*quantiles+j] = float32(-tau * invHQ)
			} else {
				quantile += (tau - 1) * e * invHQ
				grad[t*quantiles+j] = float32((1 - tau) * invHQ)
			}
		}
	}
	return mse, quantile, mse + quantile, grad, nil
}
