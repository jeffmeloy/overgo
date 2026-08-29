package evaluation

import (
	"errors"
	"math/big"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	scalingFitMediaType = "application/vnd.overgo.scaling-fit+json"
	scalingFitSchema    = "overgo/scaling-fit/v1"
)

type scalingCostAxis string

const (
	scalingCostTotalParameters  scalingCostAxis = "total-parameters"
	scalingCostActiveParameters scalingCostAxis = "active-parameters"
	scalingCostProcessedTokens  scalingCostAxis = "processed-tokens"
	scalingCostAnalyticFLOPs    scalingCostAxis = "analytic-flops"
	scalingCostMeasuredFLOPs    scalingCostAxis = "measured-flops"
	scalingCostWallNS           scalingCostAxis = "wall-ns"
)

type scalingRational struct {
	Numerator   string `json:"numerator"`
	Denominator string `json:"denominator"`
}

type scalingUncertaintyEnvelope struct {
	Lower scalingRational `json:"lower"`
	Upper scalingRational `json:"upper"`
}

type scalingFitPoint struct {
	Rung        uint32                     `json:"rung"`
	Run         artifact.ID                `json:"run"`
	Accounting  artifact.ID                `json:"accounting"`
	Evaluation  artifact.ID                `json:"evaluation"`
	Regime      artifact.ID                `json:"regime"`
	CostDecimal string                     `json:"cost_decimal"`
	Observed    scalingRational            `json:"observed"`
	Predicted   scalingRational            `json:"predicted"`
	Residual    scalingRational            `json:"residual"`
	Uncertainty scalingUncertaintyEnvelope `json:"uncertainty"`
	Covered     bool                       `json:"covered"`
}

type scalingFitAssumption struct {
	Name          string      `json:"name"`
	Evidence      artifact.ID `json:"evidence"`
	ReopenTrigger artifact.ID `json:"reopen_trigger"`
}

type scalingFit struct {
	ID                  artifact.ID            `json:"-"`
	Version             uint16                 `json:"version"`
	Study               artifact.ID            `json:"study"`
	CostAxis            scalingCostAxis        `json:"cost_axis"`
	QualityEvaluator    artifact.ID            `json:"quality_evaluator"`
	FitEvaluator        artifact.ID            `json:"fit_evaluator"`
	FitMethod           artifact.ID            `json:"fit_method"`
	UncertaintyMethod   artifact.ID            `json:"uncertainty_method"`
	UncertaintyEvidence artifact.ID            `json:"uncertainty_evidence"`
	Regime              artifact.ID            `json:"regime"`
	RegimeEvidence      artifact.ID            `json:"regime_evidence"`
	RegimeReopenTrigger artifact.ID            `json:"regime_reopen_trigger"`
	ObservedMin         string                 `json:"observed_min"`
	ObservedMax         string                 `json:"observed_max"`
	ClaimMin            string                 `json:"claim_min"`
	ClaimMax            string                 `json:"claim_max"`
	ResidualEvaluation  artifact.ID            `json:"residual_evaluation"`
	ResidualAdmitted    bool                   `json:"residual_admitted"`
	HoldoutEvaluation   artifact.ID            `json:"holdout_evaluation"`
	HoldoutAdmitted     bool                   `json:"holdout_admitted"`
	AdvisoryOnly        bool                   `json:"advisory_only"`
	FitPoints           []scalingFitPoint      `json:"fit_points"`
	Holdouts            []scalingFitPoint      `json:"holdouts"`
	Assumptions         []scalingFitAssumption `json:"assumptions"`
}

var scalingFitCodec = artifact.JSONDocumentCodec(
	"scaling fit", artifact.KindEvidence, scalingFitMediaType, scalingFitSchema,
	canonicalizeScalingFit,
	func(value scalingFit) artifact.ID { return value.ID },
	func(value *scalingFit, id artifact.ID) { value.ID = id },
	func(value scalingFit) scalingFit {
		value.FitPoints = slices.Clone(value.FitPoints)
		value.Holdouts = slices.Clone(value.Holdouts)
		value.Assumptions = slices.Clone(value.Assumptions)
		return value
	},
)

var scalingFitLineage = func(value scalingFit) []artifact.Lineage {
	parents := []artifact.ID{
		value.Study, value.QualityEvaluator, value.FitEvaluator, value.FitMethod, value.UncertaintyMethod, value.UncertaintyEvidence,
		value.Regime, value.RegimeEvidence, value.RegimeReopenTrigger,
		value.ResidualEvaluation, value.HoldoutEvaluation,
	}
	for _, point := range append(slices.Clone(value.FitPoints), value.Holdouts...) {
		parents = append(parents, point.Run, point.Accounting, point.Evaluation)
	}
	for _, assumption := range value.Assumptions {
		parents = append(parents, assumption.Evidence, assumption.ReopenTrigger)
	}
	return artifact.DependencyLineage(value.ID, uniqueArtifactIDs(parents)...)
}

func canonicalizeScalingFit(value *scalingFit) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Study.Kind() != artifact.KindEvidence ||
		value.QualityEvaluator.Kind() != artifact.KindProfile ||
		value.FitEvaluator.Kind() != artifact.KindProfile ||
		value.FitMethod.Kind() != artifact.KindProfile || value.UncertaintyMethod.Kind() != artifact.KindProfile ||
		value.UncertaintyEvidence.Kind() != artifact.KindEvidence || value.Regime.Kind() != artifact.KindProfile ||
		value.RegimeEvidence.Kind() != artifact.KindEvidence || value.RegimeReopenTrigger.Kind() != artifact.KindRecipe ||
		value.ResidualEvaluation.Kind() != artifact.KindEvidence || value.HoldoutEvaluation.Kind() != artifact.KindEvidence ||
		value.ResidualEvaluation == value.HoldoutEvaluation || value.AdvisoryOnly != (!value.ResidualAdmitted || !value.HoldoutAdmitted) ||
		!hasComparisonPair(value.FitPoints) || len(value.Holdouts) == 0 || len(value.Assumptions) == 0 {
		return errors.New("evaluation: incomplete scaling fit evidence")
	}
	switch value.CostAxis {
	case scalingCostTotalParameters, scalingCostActiveParameters, scalingCostProcessedTokens,
		scalingCostAnalyticFLOPs, scalingCostMeasuredFLOPs, scalingCostWallNS:
	default:
		return errors.New("evaluation: unknown scaling fit cost axis")
	}
	value.FitPoints = slices.Clone(value.FitPoints)
	value.Holdouts = slices.Clone(value.Holdouts)
	allPoints := append(slices.Clone(value.FitPoints), value.Holdouts...)
	seenRungs, seenRuns := make(map[uint32]bool, len(allPoints)), make(map[artifact.ID]bool, len(allPoints))
	seenAccounting, seenEvaluations := make(map[artifact.ID]bool, len(allPoints)), make(map[artifact.ID]bool, len(allPoints))
	var minimum, maximum *big.Int
	for index := range allPoints {
		point := &allPoints[index]
		cost, err := canonicalizeScalingFitPoint(point, value.Regime)
		if err != nil || seenRungs[point.Rung] || seenRuns[point.Run] || seenAccounting[point.Accounting] || seenEvaluations[point.Evaluation] {
			return errors.Join(err, errors.New("evaluation: scaling fit point is invalid, reused, or mixes regimes"))
		}
		seenRungs[point.Rung], seenRuns[point.Run] = true, true
		seenAccounting[point.Accounting], seenEvaluations[point.Evaluation] = true, true
		if minimum == nil || cost.Cmp(minimum) < 0 {
			minimum = new(big.Int).Set(cost)
		}
		if maximum == nil || cost.Cmp(maximum) > 0 {
			maximum = new(big.Int).Set(cost)
		}
	}
	if minimum.Cmp(maximum) >= 0 || value.ObservedMin != minimum.String() || value.ObservedMax != maximum.String() {
		return errors.New("evaluation: scaling fit observed range differs from exact points")
	}
	claimMin, claimMax := runrecord.ParseScalingDecimal(value.ClaimMin), runrecord.ParseScalingDecimal(value.ClaimMax)
	if claimMin == nil || claimMax == nil || claimMin.Cmp(claimMax) > 0 || claimMin.Cmp(minimum) < 0 || claimMax.Cmp(maximum) > 0 {
		return errors.New("evaluation: scaling fit claim extrapolates beyond observed evidence")
	}
	slices.SortFunc(value.FitPoints, compareScalingFitPoints)
	slices.SortFunc(value.Holdouts, compareScalingFitPoints)
	value.Assumptions = slices.Clone(value.Assumptions)
	slices.SortFunc(value.Assumptions, func(left, right scalingFitAssumption) int { return strings.Compare(left.Name, right.Name) })
	for index, assumption := range value.Assumptions {
		if assumption.Name == "" || strings.TrimSpace(assumption.Name) != assumption.Name ||
			strings.ContainsAny(assumption.Name, "\x00\r\n") || assumption.Evidence.Kind() != artifact.KindEvidence ||
			assumption.ReopenTrigger.Kind() != artifact.KindRecipe || index > 0 && value.Assumptions[index-1].Name == assumption.Name {
			return errors.New("evaluation: scaling fit assumption lacks evidence or reopen authority")
		}
	}
	return nil
}

func canonicalizeScalingFitPoint(point *scalingFitPoint, regime artifact.ID) (*big.Int, error) {
	if point == nil || point.Run.Kind() != artifact.KindRun || point.Accounting.Kind() != artifact.KindEvidence ||
		point.Evaluation.Kind() != artifact.KindEvidence || point.Regime != regime {
		return nil, errors.New("evaluation: invalid scaling fit point authorities")
	}
	cost := runrecord.ParseScalingDecimal(point.CostDecimal)
	observed, predicted := parseScalingFitRational(point.Observed), parseScalingFitRational(point.Predicted)
	residual := parseScalingFitRational(point.Residual)
	lower, upper := parseScalingFitRational(point.Uncertainty.Lower), parseScalingFitRational(point.Uncertainty.Upper)
	if cost == nil || cost.Sign() == 0 || observed == nil || predicted == nil || residual == nil || lower == nil || upper == nil || lower.Cmp(upper) > 0 {
		return nil, errors.New("evaluation: invalid scaling fit numeric evidence")
	}
	wantResidual := new(big.Rat).Sub(new(big.Rat).Set(observed), predicted)
	covered := lower.Cmp(observed) <= 0 && observed.Cmp(upper) <= 0
	if residual.Cmp(wantResidual) != 0 || point.Covered != covered {
		return nil, errors.New("evaluation: scaling residual or uncertainty coverage differs from exact values")
	}
	return cost, nil
}

func compareScalingFitPoints(left, right scalingFitPoint) int {
	comparison := runrecord.ParseScalingDecimal(left.CostDecimal).Cmp(runrecord.ParseScalingDecimal(right.CostDecimal))
	if comparison != 0 {
		return comparison
	}
	return artifact.CompareID(left.Run, right.Run)
}

func parseScalingFitRational(value scalingRational) *big.Rat {
	numerator, denominator := parseScalingFitSigned(value.Numerator), runrecord.ParseScalingDecimal(value.Denominator)
	if numerator == nil || denominator == nil || denominator.Sign() == 0 {
		return nil
	}
	parsed := new(big.Rat).SetFrac(numerator, denominator)
	if parsed.Num().String() != value.Numerator || parsed.Denom().String() != value.Denominator {
		return nil
	}
	return parsed
}

func parseScalingFitSigned(value string) *big.Int {
	if magnitudeText, found := strings.CutPrefix(value, "-"); found {
		magnitude := runrecord.ParseScalingDecimal(magnitudeText)
		if magnitude == nil || magnitude.Sign() == 0 {
			return nil
		}
		return magnitude.Neg(magnitude)
	}
	return runrecord.ParseScalingDecimal(value)
}
