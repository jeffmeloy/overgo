package evaluation

import (
	"cmp"
	"errors"
	"math/big"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	effectiveSpeedupMediaType = "application/vnd.overgo.effective-speedup+json"
	effectiveSpeedupSchema    = "overgo/effective-speedup/v1"
)

type effectiveSpeedupEndpoint struct {
	Rung        uint32      `json:"rung"`
	Run         artifact.ID `json:"run"`
	Accounting  artifact.ID `json:"accounting"`
	Evaluation  artifact.ID `json:"evaluation"`
	CostDecimal string      `json:"cost_decimal"`
}

type effectiveSpeedupPair struct {
	Quality   scalingRational          `json:"quality"`
	Baseline  effectiveSpeedupEndpoint `json:"baseline"`
	Candidate effectiveSpeedupEndpoint `json:"candidate"`
	Speedup   scalingRational          `json:"speedup"`
}

type effectiveSpeedup struct {
	ID               artifact.ID            `json:"-"`
	Version          uint16                 `json:"version"`
	BaselineFit      artifact.ID            `json:"baseline_fit"`
	CandidateFit     artifact.ID            `json:"candidate_fit"`
	CostAxis         scalingCostAxis        `json:"cost_axis"`
	QualityEvaluator artifact.ID            `json:"quality_evaluator"`
	Regime           artifact.ID            `json:"regime"`
	Pairs            []effectiveSpeedupPair `json:"pairs"`
}

var effectiveSpeedupCodec = artifact.JSONDocumentCodec(
	"effective speedup", artifact.KindEvidence, effectiveSpeedupMediaType, effectiveSpeedupSchema,
	canonicalizeEffectiveSpeedup,
	func(value effectiveSpeedup) artifact.ID { return value.ID },
	func(value *effectiveSpeedup, id artifact.ID) { value.ID = id },
	func(value effectiveSpeedup) effectiveSpeedup {
		value.Pairs = slices.Clone(value.Pairs)
		return value
	},
)

var buildEffectiveSpeedup = func(baseline, candidate scalingFit) (effectiveSpeedup, error) {
	if scalingFitCodec.ValidateIdentity(baseline) != nil || scalingFitCodec.ValidateIdentity(candidate) != nil ||
		baseline.ID == candidate.ID || baseline.CostAxis != candidate.CostAxis ||
		baseline.QualityEvaluator != candidate.QualityEvaluator || baseline.Regime != candidate.Regime {
		return effectiveSpeedup{}, errors.New("evaluation: effective speedup fit units or authorities differ")
	}
	baselinePoints := append(slices.Clone(baseline.FitPoints), baseline.Holdouts...)
	candidatePoints := append(slices.Clone(candidate.FitPoints), candidate.Holdouts...)
	pairs := make([]effectiveSpeedupPair, 0)
	for _, baselinePoint := range baselinePoints {
		baselineQuality := parseScalingFitRational(baselinePoint.Observed)
		for _, candidatePoint := range candidatePoints {
			candidateQuality := parseScalingFitRational(candidatePoint.Observed)
			if baselineQuality.Cmp(candidateQuality) != 0 {
				continue
			}
			baselineCost := runrecord.ParseScalingDecimal(baselinePoint.CostDecimal)
			candidateCost := runrecord.ParseScalingDecimal(candidatePoint.CostDecimal)
			ratio := new(big.Rat).SetFrac(baselineCost, candidateCost)
			pairs = append(pairs, effectiveSpeedupPair{
				Quality: baselinePoint.Observed,
				Baseline: effectiveSpeedupEndpoint{
					Rung: baselinePoint.Rung, Run: baselinePoint.Run, Accounting: baselinePoint.Accounting,
					Evaluation: baselinePoint.Evaluation, CostDecimal: baselinePoint.CostDecimal,
				},
				Candidate: effectiveSpeedupEndpoint{
					Rung: candidatePoint.Rung, Run: candidatePoint.Run, Accounting: candidatePoint.Accounting,
					Evaluation: candidatePoint.Evaluation, CostDecimal: candidatePoint.CostDecimal,
				},
				Speedup: scalingRational{Numerator: ratio.Num().String(), Denominator: ratio.Denom().String()},
			})
		}
	}
	if len(pairs) == 0 {
		return effectiveSpeedup{}, errors.New("evaluation: effective speedup has no exact observed quality match")
	}
	return effectiveSpeedupCodec.New(effectiveSpeedup{
		Version: artifact.InitialDocumentVersion, BaselineFit: baseline.ID, CandidateFit: candidate.ID,
		CostAxis: baseline.CostAxis, QualityEvaluator: baseline.QualityEvaluator, Regime: baseline.Regime, Pairs: pairs,
	})
}

var effectiveSpeedupLineage = func(value effectiveSpeedup) []artifact.Lineage {
	parents := []artifact.ID{value.BaselineFit, value.CandidateFit, value.QualityEvaluator, value.Regime}
	for _, pair := range value.Pairs {
		parents = append(parents,
			pair.Baseline.Run, pair.Baseline.Accounting, pair.Baseline.Evaluation,
			pair.Candidate.Run, pair.Candidate.Accounting, pair.Candidate.Evaluation,
		)
	}
	return artifact.DependencyLineage(value.ID, uniqueArtifactIDs(parents)...)
}

func canonicalizeEffectiveSpeedup(value *effectiveSpeedup) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.BaselineFit.Kind() != artifact.KindEvidence ||
		value.CandidateFit.Kind() != artifact.KindEvidence || value.BaselineFit == value.CandidateFit ||
		value.QualityEvaluator.Kind() != artifact.KindProfile || value.Regime.Kind() != artifact.KindProfile || len(value.Pairs) == 0 {
		return errors.New("evaluation: incomplete effective speedup evidence")
	}
	switch value.CostAxis {
	case scalingCostTotalParameters, scalingCostActiveParameters, scalingCostProcessedTokens,
		scalingCostAnalyticFLOPs, scalingCostMeasuredFLOPs, scalingCostWallNS:
	default:
		return errors.New("evaluation: unknown effective speedup cost axis")
	}
	value.Pairs = slices.Clone(value.Pairs)
	seen := make(map[[2]artifact.ID]bool, len(value.Pairs))
	for _, pair := range value.Pairs {
		quality, speedup := parseScalingFitRational(pair.Quality), parseScalingFitRational(pair.Speedup)
		baselineCost, candidateCost := runrecord.ParseScalingDecimal(pair.Baseline.CostDecimal), runrecord.ParseScalingDecimal(pair.Candidate.CostDecimal)
		key := [2]artifact.ID{pair.Baseline.Run, pair.Candidate.Run}
		if quality == nil || speedup == nil || baselineCost == nil || candidateCost == nil ||
			baselineCost.Sign() == 0 || candidateCost.Sign() == 0 || seen[key] ||
			!validEffectiveSpeedupEndpoint(pair.Baseline) || !validEffectiveSpeedupEndpoint(pair.Candidate) ||
			pair.Baseline.Run == pair.Candidate.Run || pair.Baseline.Accounting == pair.Candidate.Accounting ||
			pair.Baseline.Evaluation == pair.Candidate.Evaluation ||
			speedup.Cmp(new(big.Rat).SetFrac(baselineCost, candidateCost)) != 0 {
			return errors.New("evaluation: effective speedup differs from exact observed costs")
		}
		seen[key] = true
	}
	slices.SortFunc(value.Pairs, compareEffectiveSpeedupPairs)
	return nil
}

func compareEffectiveSpeedupPairs(left, right effectiveSpeedupPair) int {
	if order := compareEffectiveSpeedupRational(left.Quality, right.Quality); order != 0 {
		return order
	}
	if order := compareEffectiveSpeedupEndpoints(left.Baseline, right.Baseline); order != 0 {
		return order
	}
	if order := compareEffectiveSpeedupEndpoints(left.Candidate, right.Candidate); order != 0 {
		return order
	}
	return compareEffectiveSpeedupRational(left.Speedup, right.Speedup)
}

func compareEffectiveSpeedupEndpoints(left, right effectiveSpeedupEndpoint) int {
	if order := artifact.CompareID(left.Run, right.Run); order != 0 {
		return order
	}
	if order := cmp.Compare(left.Rung, right.Rung); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.Accounting, right.Accounting); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.Evaluation, right.Evaluation); order != 0 {
		return order
	}
	leftCost, rightCost := runrecord.ParseScalingDecimal(left.CostDecimal), runrecord.ParseScalingDecimal(right.CostDecimal)
	if leftCost != nil && rightCost != nil {
		if order := leftCost.Cmp(rightCost); order != 0 {
			return order
		}
	}
	return cmp.Compare(left.CostDecimal, right.CostDecimal)
}

func compareEffectiveSpeedupRational(left, right scalingRational) int {
	leftValue, rightValue := parseScalingFitRational(left), parseScalingFitRational(right)
	if leftValue != nil && rightValue != nil {
		if order := leftValue.Cmp(rightValue); order != 0 {
			return order
		}
	}
	if order := cmp.Compare(left.Numerator, right.Numerator); order != 0 {
		return order
	}
	return cmp.Compare(left.Denominator, right.Denominator)
}

func validEffectiveSpeedupEndpoint(value effectiveSpeedupEndpoint) bool {
	return value.Run.Kind() == artifact.KindRun && value.Accounting.Kind() == artifact.KindEvidence &&
		value.Evaluation.Kind() == artifact.KindEvidence
}
