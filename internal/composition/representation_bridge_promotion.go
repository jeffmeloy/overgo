package composition

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
)

const (
	RepresentationBridgePromotionVersion   = artifact.InitialDocumentVersion
	RepresentationBridgePromotionMediaType = "application/vnd.overgo.representation-bridge-promotion+json"
	RepresentationBridgePromotionSchema    = "overgo/representation-bridge-promotion/v1"
)

// RepresentationBridgePromotionPolicy declares one metric and the minimum
// evidence envelope that every independent seed must satisfy.
type RepresentationBridgePromotionPolicy struct {
	Direction               runrecord.Direction `json:"direction"`
	MinimumSeeds            uint64              `json:"minimum_seeds"`
	MinimumHeldOutGain      float64             `json:"minimum_held_out_gain"`
	MinimumSourceDependence float64             `json:"minimum_source_dependence"`
	MaximumRegression       float64             `json:"maximum_regression"`
}

// RepresentationBridgePromotionTrial records the same held-out metric for a
// trained bridge, its cheap baseline, two source ablations, and a target-only
// regression control. Scores are interpreted using the policy direction.
type RepresentationBridgePromotionTrial struct {
	Seed                     uint64  `json:"seed"`
	BridgeScore              float64 `json:"bridge_score"`
	CheapBaselineScore       float64 `json:"cheap_baseline_score"`
	DroppedSourceScore       float64 `json:"dropped_source_score"`
	ShuffledSourceScore      float64 `json:"shuffled_source_score"`
	RegressionBaselineScore  float64 `json:"regression_baseline_score"`
	RegressionCandidateScore float64 `json:"regression_candidate_score"`
}

// RepresentationBridgePromotion is immutable evidence that a particular
// bridge cleared its complete promotion envelope on an exact held-out split.
// The worst margins are derived during canonicalization, never caller-trusted.
type RepresentationBridgePromotion struct {
	Version               uint16                               `json:"version"`
	Bridge                artifact.ID                          `json:"bridge"`
	SourceModel           artifact.ID                          `json:"source_model"`
	TargetModel           artifact.ID                          `json:"target_model"`
	SourceContract        artifact.ID                          `json:"source_contract"`
	TargetContract        artifact.ID                          `json:"target_contract"`
	HeldOutSplit          artifact.ID                          `json:"held_out_split"`
	RegressionSet         artifact.ID                          `json:"regression_set"`
	Evaluator             artifact.ID                          `json:"evaluator"`
	Policy                RepresentationBridgePromotionPolicy  `json:"policy"`
	Trials                []RepresentationBridgePromotionTrial `json:"trials"`
	WorstHeldOutGain      float64                              `json:"worst_held_out_gain"`
	WorstSourceDependence float64                              `json:"worst_source_dependence"`
	WorstRegression       float64                              `json:"worst_regression"`
	ID                    artifact.ID                          `json:"-"`
}

// RepresentationBridgePromoter validates evidence but has no authority to
// train, load, or mutate either model.
type RepresentationBridgePromoter struct{}

var representationBridgePromotionCodec = artifact.JSONDocumentCodec(
	"representation bridge promotion",
	artifact.KindEvidence,
	RepresentationBridgePromotionMediaType,
	RepresentationBridgePromotionSchema,
	canonicalizeRepresentationBridgePromotion,
	func(value RepresentationBridgePromotion) artifact.ID { return value.ID },
	func(value *RepresentationBridgePromotion, id artifact.ID) { value.ID = id },
	func(value RepresentationBridgePromotion) RepresentationBridgePromotion {
		value.Trials = slices.Clone(value.Trials)
		return value
	},
)

// Evaluate returns promotion evidence only when every seed clears every
// declared threshold. A partial or average-only win is refused.
func (RepresentationBridgePromoter) Evaluate(
	value RepresentationBridgePromotion,
) (RepresentationBridgePromotion, error) {
	value.Version = RepresentationBridgePromotionVersion
	return representationBridgePromotionCodec.New(value)
}

// Parse validates canonical serialized promotion evidence.
func (RepresentationBridgePromoter) Parse(data []byte) (RepresentationBridgePromotion, error) {
	return representationBridgePromotionCodec.Parse(data)
}

// ValidateIdentity verifies both the evidence envelope and content identity.
func (value RepresentationBridgePromotion) ValidateIdentity() error {
	return representationBridgePromotionCodec.ValidateIdentity(value)
}

// Content returns the exact promotion evidence document.
func (value RepresentationBridgePromotion) Content() (artifact.Content, error) {
	return representationBridgePromotionCodec.Content(value)
}

// Lineage binds the decision to the bridge, models, representation contracts,
// evaluation partitions, and evaluator authority it actually measured.
func (value RepresentationBridgePromotion) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(
		value.ID,
		value.Bridge,
		value.SourceModel,
		value.TargetModel,
		value.SourceContract,
		value.TargetContract,
		value.HeldOutSplit,
		value.RegressionSet,
		value.Evaluator,
	)
}

// Batch publishes the evidence and all exact dependency edges atomically.
func (value RepresentationBridgePromotion) Batch(key string) (artifact.Batch, error) {
	return representationBridgePromotionCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeRepresentationBridgePromotion(value *RepresentationBridgePromotion) error {
	if value == nil || value.Version != RepresentationBridgePromotionVersion {
		return errors.New("composition: invalid representation bridge promotion version")
	}
	if value.Bridge.Kind() != artifact.KindAdapter ||
		value.SourceModel.Kind() != artifact.KindModel ||
		value.TargetModel.Kind() != artifact.KindModel ||
		value.SourceModel == value.TargetModel ||
		value.SourceContract.Kind() != artifact.KindProfile ||
		value.TargetContract.Kind() != artifact.KindProfile ||
		value.HeldOutSplit.Kind() != artifact.KindDatasetShard ||
		value.RegressionSet.Kind() != artifact.KindDatasetShard ||
		value.Evaluator.Kind() != artifact.KindEvidence {
		return errors.New("composition: representation bridge promotion authorities are invalid")
	}
	if err := value.Policy.validate(); err != nil {
		return err
	}
	if uint64(len(value.Trials)) < value.Policy.MinimumSeeds {
		return errors.New("composition: representation bridge promotion lacks independent seeds")
	}
	sort.Slice(value.Trials, func(left, right int) bool {
		return value.Trials[left].Seed < value.Trials[right].Seed
	})
	first := value.Trials[tensor.FirstOffset]
	if err := first.validate(); err != nil {
		return err
	}
	worstGain, sourceDependence, regression := value.Policy.margins(first)
	if err := value.Policy.admit(worstGain, sourceDependence, regression); err != nil {
		return err
	}
	priorSeed := first.Seed
	for _, trial := range value.Trials[tensor.SingletonExtent:] {
		if err := trial.validate(); err != nil {
			return err
		}
		if trial.Seed == priorSeed {
			return errors.New("composition: representation bridge promotion seeds are not independent")
		}
		priorSeed = trial.Seed
		gain, dependence, trialRegression := value.Policy.margins(trial)
		if err := value.Policy.admit(gain, dependence, trialRegression); err != nil {
			return err
		}
		worstGain = min(worstGain, gain)
		sourceDependence = min(sourceDependence, dependence)
		regression = max(regression, trialRegression)
	}
	value.WorstHeldOutGain = worstGain
	value.WorstSourceDependence = sourceDependence
	value.WorstRegression = regression
	return nil
}

func (policy RepresentationBridgePromotionPolicy) validate() error {
	if policy.Direction != runrecord.DirectionMinimize && policy.Direction != runrecord.DirectionMaximize ||
		policy.MinimumSeeds < uint64(tensor.PairedExtent) ||
		!checked.PositiveFinite64(policy.MinimumHeldOutGain) ||
		!checked.PositiveFinite64(policy.MinimumSourceDependence) ||
		!checked.Finite64(policy.MaximumRegression) ||
		policy.MaximumRegression < float64(tensor.FirstOffset) {
		return errors.New("composition: invalid representation bridge promotion policy")
	}
	return nil
}

func (trial RepresentationBridgePromotionTrial) validate() error {
	values := [...]float64{
		trial.BridgeScore,
		trial.CheapBaselineScore,
		trial.DroppedSourceScore,
		trial.ShuffledSourceScore,
		trial.RegressionBaselineScore,
		trial.RegressionCandidateScore,
	}
	for _, value := range values {
		if !checked.Finite64(value) {
			return errors.New("composition: representation bridge promotion score is non-finite")
		}
	}
	return nil
}

func (policy RepresentationBridgePromotionPolicy) margins(
	trial RepresentationBridgePromotionTrial,
) (gain, dependence, regression float64) {
	if policy.Direction == runrecord.DirectionMaximize {
		gain = trial.BridgeScore - trial.CheapBaselineScore
		dependence = min(
			trial.BridgeScore-trial.DroppedSourceScore,
			trial.BridgeScore-trial.ShuffledSourceScore,
		)
		regression = trial.RegressionBaselineScore - trial.RegressionCandidateScore
		return gain, dependence, regression
	}
	gain = trial.CheapBaselineScore - trial.BridgeScore
	dependence = min(
		trial.DroppedSourceScore-trial.BridgeScore,
		trial.ShuffledSourceScore-trial.BridgeScore,
	)
	regression = trial.RegressionCandidateScore - trial.RegressionBaselineScore
	return gain, dependence, regression
}

func (policy RepresentationBridgePromotionPolicy) admit(gain, dependence, regression float64) error {
	if !checked.Finite64(gain) || !checked.Finite64(dependence) || !checked.Finite64(regression) {
		return errors.New("composition: representation bridge promotion margin is non-finite")
	}
	if gain < policy.MinimumHeldOutGain {
		return errors.New("composition: representation bridge lacks held-out gain")
	}
	if dependence < policy.MinimumSourceDependence {
		return errors.New("composition: representation bridge lacks source dependence")
	}
	if regression > policy.MaximumRegression {
		return errors.New("composition: representation bridge exceeds regression allowance")
	}
	return nil
}
