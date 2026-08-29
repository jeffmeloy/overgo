package densecausal

import (
	"errors"
	"math"

	"overgo/internal/artifact"
)

type moeQuantileBiasPolicy struct {
	Stratum, Authority artifact.ID
	Target             moeQuantileSpec
}

type moeQuantileBiasController struct {
	policy   moeQuantileBiasPolicy
	nextBias []float32
}

// routeTopKWithQuantileController applies only the bias derived from the prior
// call, then derives replacement state from this call's unbiased activated
// scores. Approximate quantiles cannot silently acquire controller authority.
var routeTopKWithQuantileController = func(
	x, router []float32,
	rows, hidden, experts int,
	routerPolicy MoERouterPolicy,
	stratum artifact.ID,
	controller *moeQuantileBiasController,
) (moeRoute, []float32, error) {
	if controller == nil || controller.policy.Stratum.Kind() != artifact.KindDatasetShard || controller.policy.Authority.Kind() != artifact.KindRecipe ||
		stratum != controller.policy.Stratum || controller.policy.Target.Method != moeQuantileExact ||
		controller.policy.Target.Bins != 0 || len(controller.nextBias) != 0 && len(controller.nextBias) != experts {
		return moeRoute{}, nil, errors.New("densecausal: invalid or incompatible quantile-bias controller policy")
	}
	route, scores, err := routeTopKSelectionBias(x, router, rows, hidden, experts, routerPolicy, controller.nextBias)
	if err != nil {
		return moeRoute{}, nil, err
	}
	expertSamples := make([][]float64, experts)
	for expert := range experts {
		expertSamples[expert] = make([]float64, rows)
		for row := range rows {
			expertSamples[expert][row] = float64(scores[row*experts+expert])
		}
	}
	evidence, err := estimateMoEQuantiles(controller.policy.Target, []moeQuantileStratum{{
		Stratum: stratum, ExpertSamples: expertSamples,
	}})
	if err != nil || !evidence.Complete {
		return moeRoute{}, nil, errors.Join(err, errors.New("densecausal: quantile-bias update lacks complete exact evidence"))
	}
	thresholds := make([]float64, experts)
	minimum, maximum := math.Inf(1), math.Inf(-1)
	for expert, estimate := range evidence.Strata[0].Experts {
		if estimate.Failure != "" || estimate.ValueLower != estimate.ValueUpper || estimate.MaximumRankError != 0 {
			return moeRoute{}, nil, errors.New("densecausal: quantile-bias update requires point-valued exact thresholds")
		}
		thresholds[expert] = estimate.ValueLower
		minimum, maximum = min(minimum, estimate.ValueLower), max(maximum, estimate.ValueUpper)
	}
	// A common additive offset cannot change expert ranking. Midrange centering
	// therefore fixes that gauge while bounding every bias by half the observed
	// threshold range, without a learned scale or clipping hyperparameter.
	offset := minimum + (maximum-minimum)/2
	next := make([]float32, experts)
	for expert, threshold := range thresholds {
		next[expert] = float32(offset - threshold)
	}
	controller.nextBias = next
	return route, scores, nil
}
