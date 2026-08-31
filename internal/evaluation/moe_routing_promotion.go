package evaluation

import (
	"errors"

	"overgo/internal/artifact"
)

const (
	moeRoutingPromotionPolicyMediaType = "application/vnd.overgo.moe-routing-promotion-policy+json"
	moeRoutingPromotionPolicySchema    = "overgo/moe-routing-promotion-policy/v1"
	moeRoutingPromotionMediaType       = "application/vnd.overgo.moe-routing-promotion+json"
	moeRoutingPromotionSchema          = "overgo/moe-routing-promotion/v1"
)

type moeRoutingComparisonOutcome string

const (
	moeRoutingEquivalent moeRoutingComparisonOutcome = "equivalent"
	moeRoutingImproved   moeRoutingComparisonOutcome = "improved"
)

type moeRoutingPromotionPolicy struct {
	ID                   artifact.ID `json:"-"`
	Version              uint16      `json:"version"`
	QualityEvaluator     artifact.ID `json:"quality_evaluator"`
	StabilityEvaluator   artifact.ID `json:"stability_evaluator"`
	UtilizationEvaluator artifact.ID `json:"utilization_evaluator"`
	DropEvaluator        artifact.ID `json:"drop_evaluator"`
	CostEvaluator        artifact.ID `json:"cost_evaluator"`
}

type moeRoutingPromotionComparison struct {
	Evaluator artifact.ID                 `json:"evaluator"`
	Evidence  artifact.ID                 `json:"evidence"`
	Outcome   moeRoutingComparisonOutcome `json:"outcome"`
}

type moeRoutingPromotionComparisons struct {
	Quality     moeRoutingPromotionComparison `json:"quality"`
	Stability   moeRoutingPromotionComparison `json:"stability"`
	Utilization moeRoutingPromotionComparison `json:"utilization"`
	Drop        moeRoutingPromotionComparison `json:"drop"`
	Cost        moeRoutingPromotionComparison `json:"cost"`
}

type moeRoutingPromotion struct {
	ID             artifact.ID `json:"-"`
	Version        uint16      `json:"version"`
	Policy         artifact.ID `json:"policy"`
	Candidate      artifact.ID `json:"candidate"`
	BaselineMatrix artifact.ID `json:"baseline_matrix"`
	ShiftEvidence  artifact.ID `json:"shift_evidence"`
	CapacityPair   artifact.ID `json:"capacity_pair"`
	moeRoutingPromotionComparisons
}

var moeRoutingPromotionPolicyCodec = artifact.JSONDocumentCodec(
	"MoE routing promotion policy", artifact.KindProfile,
	moeRoutingPromotionPolicyMediaType, moeRoutingPromotionPolicySchema,
	func(value *moeRoutingPromotionPolicy) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion {
			return errors.New("evaluation: invalid MoE routing promotion policy")
		}
		evaluators := [...]artifact.ID{
			value.QualityEvaluator, value.StabilityEvaluator, value.UtilizationEvaluator, value.DropEvaluator, value.CostEvaluator,
		}
		seen := make(map[artifact.ID]bool, len(evaluators))
		for _, evaluator := range evaluators {
			if evaluator.Kind() != artifact.KindProfile || seen[evaluator] {
				return errors.New("evaluation: promotion dimensions require distinct evaluator authorities")
			}
			seen[evaluator] = true
		}
		return nil
	},
	func(value moeRoutingPromotionPolicy) artifact.ID { return value.ID },
	func(value *moeRoutingPromotionPolicy, id artifact.ID) { value.ID = id }, nil,
)

var moeRoutingPromotionCodec = artifact.JSONDocumentCodec(
	"MoE routing promotion", artifact.KindEvidence, moeRoutingPromotionMediaType, moeRoutingPromotionSchema,
	canonicalizeMoERoutingPromotion,
	func(value moeRoutingPromotion) artifact.ID { return value.ID },
	func(value *moeRoutingPromotion, id artifact.ID) { value.ID = id }, nil,
)

var moeRoutingPromotionPolicyLineage = func(value moeRoutingPromotionPolicy) []artifact.Lineage {
	return artifact.DependencyLineage(value.ID,
		value.QualityEvaluator, value.StabilityEvaluator, value.UtilizationEvaluator, value.DropEvaluator, value.CostEvaluator,
	)
}

var buildMoERoutingPromotion = func(
	policy moeRoutingPromotionPolicy,
	baseline moeRoutingBaselineMatrix,
	shift moeRoutingShiftEvidence,
	capacity moeCapacityPair,
	candidate artifact.ID,
	comparisons moeRoutingPromotionComparisons,
) (moeRoutingPromotion, error) {
	if moeRoutingPromotionPolicyCodec.ValidateIdentity(policy) != nil || moeRoutingBaselineCodec.ValidateIdentity(baseline) != nil ||
		moeRoutingShiftCodec.ValidateIdentity(shift) != nil || moeCapacityPairCodec.ValidateIdentity(capacity) != nil ||
		shift.BaselineMatrix != baseline.ID || shift.HiddenStratumFailure || shift.ControllerLag || capacity.DropQualityGap ||
		!capacity.Constrained.Admitted || !capacity.Dropless.Admitted || capacity.Constrained.Policy != candidate ||
		capacity.Model != baseline.Model || capacity.Dataset != baseline.Dataset || capacity.Split != baseline.Split ||
		capacity.Checkpoint != baseline.Checkpoint || capacity.Seed != baseline.Seed || capacity.Code != baseline.Code ||
		capacity.Environment != baseline.Environment || capacity.Evaluator != baseline.Evaluator {
		return moeRoutingPromotion{}, errors.New("evaluation: MoE routing promotion authorities or paired evidence differ")
	}
	quantileCandidate := false
	for _, arm := range baseline.Arms {
		if arm.Kind == moeRoutingQuantileBias && arm.Recipe == candidate {
			quantileCandidate = true
		}
	}
	if !quantileCandidate || comparisons.Quality.Evaluator != policy.QualityEvaluator ||
		comparisons.Stability.Evaluator != policy.StabilityEvaluator ||
		comparisons.Utilization.Evaluator != policy.UtilizationEvaluator ||
		comparisons.Drop.Evaluator != policy.DropEvaluator || comparisons.Cost.Evaluator != policy.CostEvaluator {
		return moeRoutingPromotion{}, errors.New("evaluation: MoE routing promotion policy differs")
	}
	return moeRoutingPromotionCodec.New(moeRoutingPromotion{
		Version: artifact.InitialDocumentVersion, Policy: policy.ID, Candidate: candidate,
		BaselineMatrix: baseline.ID, ShiftEvidence: shift.ID, CapacityPair: capacity.ID,
		moeRoutingPromotionComparisons: comparisons,
	})
}

var moeRoutingPromotionLineage = func(value moeRoutingPromotion) []artifact.Lineage {
	return artifact.DependencyLineage(value.ID,
		value.Policy, value.Candidate, value.BaselineMatrix, value.ShiftEvidence, value.CapacityPair,
		value.Quality.Evidence, value.Stability.Evidence, value.Utilization.Evidence, value.Drop.Evidence, value.Cost.Evidence,
	)
}

func canonicalizeMoERoutingPromotion(value *moeRoutingPromotion) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Policy.Kind() != artifact.KindProfile ||
		value.Candidate.Kind() != artifact.KindRecipe || value.BaselineMatrix.Kind() != artifact.KindRecipe ||
		value.ShiftEvidence.Kind() != artifact.KindEvidence || value.CapacityPair.Kind() != artifact.KindEvidence {
		return errors.New("evaluation: invalid MoE routing promotion")
	}
	comparisons := [...]moeRoutingPromotionComparison{value.Quality, value.Stability, value.Utilization, value.Drop, value.Cost}
	seenEvidence := make(map[artifact.ID]bool, len(comparisons))
	improved := false
	for _, comparison := range comparisons {
		if comparison.Evaluator.Kind() != artifact.KindProfile || comparison.Evidence.Kind() != artifact.KindEvidence ||
			seenEvidence[comparison.Evidence] ||
			(comparison.Outcome != moeRoutingEquivalent && comparison.Outcome != moeRoutingImproved) {
			return errors.New("evaluation: MoE routing promotion has missing, reused, or regressed evidence")
		}
		seenEvidence[comparison.Evidence] = true
		improved = improved || comparison.Outcome == moeRoutingImproved
	}
	if !improved {
		return errors.New("evaluation: MoE routing promotion has no strict improvement")
	}
	return nil
}
