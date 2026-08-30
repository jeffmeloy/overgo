package composition

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// CompositeDimension is one measured fitness dimension: the baseline and
// candidate values on the target held-out eval, and the direction in which
// the candidate must not regress.
type CompositeDimension struct {
	Baseline  float64             `json:"baseline"`
	Candidate float64             `json:"candidate"`
	Direction runrecord.Direction `json:"direction"`
}

// CompositeScore is one realized composite measured on the target held-out
// eval across the multidimensional improvement fitness, citing the
// evaluation evidence behind the measurements.
type CompositeScore struct {
	Composite  artifact.ID                   `json:"composite"`
	Evaluation artifact.ID                   `json:"evaluation"`
	Dimensions map[string]CompositeDimension `json:"dimensions"`
}

// CompositeFitnessVerdict is the unscalarized verdict for one composite: a
// composite improves only when no dimension regresses and at least one
// strictly improves — an average-only win cannot pass by construction,
// because the dimensions are never blended.
type CompositeFitnessVerdict struct {
	Composite  artifact.ID `json:"composite"`
	Evaluation artifact.ID `json:"evaluation"`
	Improved   []string    `json:"improved,omitempty"`
	Regressed  []string    `json:"regressed,omitempty"`
	Fit        bool        `json:"fit"`
}

// ScoreCompositeSelection judges realized composites against the
// multidimensional improvement fitness without scalarization: each named
// dimension compares in its own direction, the verdict lists exactly which
// dimensions improved and which regressed, and a composite is fit only
// when nothing regressed and something strictly improved. Verdicts order
// deterministically — fit composites first, more improved dimensions
// first, then canonical identity.
func ScoreCompositeSelection(scores []CompositeScore) ([]CompositeFitnessVerdict, error) {
	if len(scores) == 0 {
		return nil, errors.New("composition: fitness selection requires scored composites")
	}
	verdicts := make([]CompositeFitnessVerdict, 0, len(scores))
	for _, score := range scores {
		if score.Composite.Kind() != artifact.KindModel || score.Evaluation.Kind() != artifact.KindEvidence {
			return nil, errors.New("composition: a score requires exact composite and evaluation identities")
		}
		if len(score.Dimensions) == 0 {
			return nil, fmt.Errorf("composition: composite %s was scored on no dimensions", score.Composite)
		}
		verdict := CompositeFitnessVerdict{Composite: score.Composite, Evaluation: score.Evaluation}
		names := make([]string, 0, len(score.Dimensions))
		for name := range score.Dimensions {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			dimension := score.Dimensions[name]
			if math.IsNaN(dimension.Baseline) || math.IsInf(dimension.Baseline, 0) ||
				math.IsNaN(dimension.Candidate) || math.IsInf(dimension.Candidate, 0) {
				return nil, fmt.Errorf("composition: dimension %q of %s is not finite", name, score.Composite)
			}
			delta := dimension.Candidate - dimension.Baseline
			if dimension.Direction == runrecord.DirectionMinimize {
				delta = -delta
			} else if dimension.Direction != runrecord.DirectionMaximize {
				return nil, fmt.Errorf("composition: dimension %q of %s has no comparison direction", name, score.Composite)
			}
			switch {
			case delta > 0:
				verdict.Improved = append(verdict.Improved, name)
			case delta < 0:
				verdict.Regressed = append(verdict.Regressed, name)
			}
		}
		verdict.Fit = len(verdict.Regressed) == 0 && len(verdict.Improved) > 0
		verdicts = append(verdicts, verdict)
	}
	slices.SortFunc(verdicts, func(a, b CompositeFitnessVerdict) int {
		if a.Fit != b.Fit {
			if a.Fit {
				return -1
			}
			return 1
		}
		if len(a.Improved) != len(b.Improved) {
			return len(b.Improved) - len(a.Improved)
		}
		return strings.Compare(a.Composite.String(), b.Composite.String())
	})
	return verdicts, nil
}

// EmitAblationGatedPromotion emits one fit composite through the existing
// ablation-armed composition promotion gate: only a composite the
// multidimensional fitness judged fit may enter, and the promotion
// evidence — with its dropped-source and shuffled-source arms — validates
// entirely inside the existing RepresentationBridgePromoter, so this adds
// no second promotion path.
func EmitAblationGatedPromotion(
	verdict CompositeFitnessVerdict,
	policy RepresentationBridgePromotionPolicy,
	evidence RepresentationBridgePromotion,
) (RepresentationBridgePromotion, error) {
	if !verdict.Fit {
		return RepresentationBridgePromotion{}, fmt.Errorf(
			"composition: composite %s did not clear the multidimensional fitness; regressed=%v",
			verdict.Composite, verdict.Regressed,
		)
	}
	return RepresentationBridgePromoter{}.Evaluate(policy, evidence)
}
