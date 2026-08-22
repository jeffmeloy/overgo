package composition

import (
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
)

const (
	// RepresentationBridgePromotionVersion is the current promotion document version.
	RepresentationBridgePromotionVersion = artifact.InitialDocumentVersion
	// RepresentationBridgePromotionMediaType identifies serialized bridge-promotion evidence.
	RepresentationBridgePromotionMediaType = "application/vnd.overgo.representation-bridge-promotion+json"
	// RepresentationBridgePromotionSchema identifies the canonical bridge-promotion schema.
	RepresentationBridgePromotionSchema = "overgo/representation-bridge-promotion/v1"
	// RepresentationBridgePromotionPolicyMediaType identifies recipe-owned promotion policy.
	RepresentationBridgePromotionPolicyMediaType = "application/vnd.overgo.representation-bridge-promotion-policy+json"
	// RepresentationBridgePromotionPolicySchema identifies the policy wire schema.
	RepresentationBridgePromotionPolicySchema = "overgo/representation-bridge-promotion-policy/v1"
)

// RepresentationBridgePromotionPolicy declares one metric and the minimum
// evidence envelope that every independent seed must satisfy.
type RepresentationBridgePromotionPolicy struct {
	Version                   uint16              `json:"version"`
	Direction                 runrecord.Direction `json:"direction"`
	MinimumSeeds              uint64              `json:"minimum_seeds"`
	MinimumHeldOutGain        float64             `json:"minimum_held_out_gain"`
	MinimumSourceDependence   float64             `json:"minimum_source_dependence"`
	MaximumRegression         float64             `json:"maximum_regression"`
	MaximumSeedSpread         float64             `json:"maximum_seed_spread"`
	MaximumLatencyIncrease    float64             `json:"maximum_latency_increase"`
	MaximumDeviceByteIncrease uint64              `json:"maximum_device_byte_increase"`
	ID                        artifact.ID         `json:"-"`
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
	BaselineLatencyNS        uint64  `json:"baseline_latency_ns"`
	ComposedLatencyNS        uint64  `json:"composed_latency_ns"`
	BaselinePeakDeviceBytes  uint64  `json:"baseline_peak_device_bytes"`
	ComposedPeakDeviceBytes  uint64  `json:"composed_peak_device_bytes"`
}

// RepresentationBridgePromotion is immutable evidence that a particular
// bridge cleared its complete promotion envelope on an exact held-out split.
// The worst margins are derived during canonicalization, never caller-trusted.
type RepresentationBridgePromotion struct {
	Version                 uint16                               `json:"version"`
	Bridge                  artifact.ID                          `json:"bridge"`
	SourceModel             artifact.ID                          `json:"source_model"`
	TargetModel             artifact.ID                          `json:"target_model"`
	SourceContract          artifact.ID                          `json:"source_contract"`
	TargetContract          artifact.ID                          `json:"target_contract"`
	HeldOutSplit            artifact.ID                          `json:"held_out_split"`
	RegressionSet           artifact.ID                          `json:"regression_set"`
	Evaluator               artifact.ID                          `json:"evaluator"`
	PolicyID                artifact.ID                          `json:"policy_id"`
	Policy                  RepresentationBridgePromotionPolicy  `json:"policy"`
	Trials                  []RepresentationBridgePromotionTrial `json:"trials"`
	WorstHeldOutGain        float64                              `json:"worst_held_out_gain"`
	WorstSourceDependence   float64                              `json:"worst_source_dependence"`
	WorstRegression         float64                              `json:"worst_regression"`
	SeedSpread              float64                              `json:"seed_spread"`
	WorstLatencyIncrease    float64                              `json:"worst_latency_increase"`
	WorstDeviceByteIncrease uint64                               `json:"worst_device_byte_increase"`
	ID                      artifact.ID                          `json:"-"`
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

var representationBridgePromotionPolicyCodec = artifact.JSONDocumentCodec(
	"representation bridge promotion policy", artifact.KindProfile,
	RepresentationBridgePromotionPolicyMediaType, RepresentationBridgePromotionPolicySchema,
	func(value *RepresentationBridgePromotionPolicy) error { return value.validate() },
	func(value RepresentationBridgePromotionPolicy) artifact.ID { return value.ID },
	func(value *RepresentationBridgePromotionPolicy, id artifact.ID) { value.ID = id },
	func(value RepresentationBridgePromotionPolicy) RepresentationBridgePromotionPolicy { return value },
)

// NewRepresentationBridgePromotionPolicy validates and identifies recipe-owned thresholds.
func NewRepresentationBridgePromotionPolicy(
	value RepresentationBridgePromotionPolicy,
) (RepresentationBridgePromotionPolicy, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return representationBridgePromotionPolicyCodec.New(value)
}

// Content returns the exact recipe-owned promotion policy document.
func (policy RepresentationBridgePromotionPolicy) Content() (artifact.Content, error) {
	return representationBridgePromotionPolicyCodec.Content(policy)
}

// ValidateIdentity verifies the immutable promotion policy identity.
func (policy RepresentationBridgePromotionPolicy) ValidateIdentity() error {
	return representationBridgePromotionPolicyCodec.ValidateIdentity(policy)
}

// LoadRepresentationBridgePromotionPolicy resolves exact recipe-owned thresholds.
func LoadRepresentationBridgePromotionPolicy(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (RepresentationBridgePromotionPolicy, error) {
	return representationBridgePromotionPolicyCodec.Require(ctx, reader, id)
}

// Identity returns the exact recipe-owned promotion-policy identity.
func (policy RepresentationBridgePromotionPolicy) Identity() (artifact.ID, error) {
	if err := policy.ValidateIdentity(); err != nil {
		return artifact.ID{}, err
	}
	return policy.ID, nil
}

func (policy RepresentationBridgePromotionPolicy) content() (artifact.Content, error) {
	return policy.Content()
}

func loadRepresentationBridgePromotionPolicy(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (RepresentationBridgePromotionPolicy, error) {
	return LoadRepresentationBridgePromotionPolicy(ctx, reader, id)
}

// Evaluate returns promotion evidence only when every seed clears every
// declared threshold. A partial or average-only win is refused.
func (RepresentationBridgePromoter) Evaluate(
	policy RepresentationBridgePromotionPolicy,
	value RepresentationBridgePromotion,
) (RepresentationBridgePromotion, error) {
	if err := policy.ValidateIdentity(); err != nil {
		return RepresentationBridgePromotion{}, err
	}
	value.Version = RepresentationBridgePromotionVersion
	value.PolicyID, value.Policy = policy.ID, policy
	return representationBridgePromotionCodec.New(value)
}

// Parse validates canonical serialized promotion evidence.
func (RepresentationBridgePromoter) Parse(data []byte) (RepresentationBridgePromotion, error) {
	return representationBridgePromotionCodec.Parse(data)
}

// LoadRepresentationBridgePromotion resolves exact immutable promotion
// evidence and rejects descriptor, schema, or content-identity drift.
func LoadRepresentationBridgePromotion(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (RepresentationBridgePromotion, error) {
	content, err := loadCompositionContent(
		ctx, reader, id, artifact.KindEvidence,
		RepresentationBridgePromotionMediaType, RepresentationBridgePromotionSchema,
	)
	if err != nil {
		return RepresentationBridgePromotion{}, err
	}
	value, err := (RepresentationBridgePromoter{}).Parse(content.Data)
	if err != nil || value.ID != id {
		return RepresentationBridgePromotion{}, errors.Join(
			err, errors.New("composition: representation bridge promotion identity differs"),
		)
	}
	return value, nil
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
		value.PolicyID,
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
	policy, err := NewRepresentationBridgePromotionPolicy(value.Policy)
	if err != nil || value.PolicyID != policy.ID {
		return errors.Join(err, errors.New("composition: representation bridge promotion policy identity differs"))
	}
	value.Policy = policy
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
	latency, deviceBytes, err := value.Policy.resourceMargins(first)
	if err != nil {
		return err
	}
	minimumScore, maximumScore := first.BridgeScore, first.BridgeScore
	if err := value.Policy.admit(worstGain, sourceDependence, regression, latency, deviceBytes); err != nil {
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
		trialLatency, trialDeviceBytes, resourceErr := value.Policy.resourceMargins(trial)
		if resourceErr != nil {
			return resourceErr
		}
		if err := value.Policy.admit(gain, dependence, trialRegression, trialLatency, trialDeviceBytes); err != nil {
			return err
		}
		worstGain = min(worstGain, gain)
		sourceDependence = min(sourceDependence, dependence)
		regression = max(regression, trialRegression)
		latency = max(latency, trialLatency)
		deviceBytes = max(deviceBytes, trialDeviceBytes)
		minimumScore, maximumScore = min(minimumScore, trial.BridgeScore), max(maximumScore, trial.BridgeScore)
	}
	spread := maximumScore - minimumScore
	if spread > value.Policy.MaximumSeedSpread {
		return errors.New("composition: representation bridge repeated-run stability is insufficient")
	}
	value.WorstHeldOutGain = worstGain
	value.WorstSourceDependence = sourceDependence
	value.WorstRegression = regression
	value.SeedSpread, value.WorstLatencyIncrease = spread, latency
	value.WorstDeviceByteIncrease = deviceBytes
	return nil
}

func (policy RepresentationBridgePromotionPolicy) validate() error {
	if policy.Version != artifact.InitialDocumentVersion ||
		policy.Direction != runrecord.DirectionMinimize && policy.Direction != runrecord.DirectionMaximize ||
		policy.MinimumSeeds < uint64(tensor.PairedExtent) ||
		!checked.PositiveFinite64(policy.MinimumHeldOutGain) ||
		!checked.PositiveFinite64(policy.MinimumSourceDependence) ||
		!checked.Finite64(policy.MaximumRegression) ||
		policy.MaximumRegression < float64(tensor.FirstOffset) ||
		!checked.PositiveFinite64(policy.MaximumSeedSpread) ||
		!checked.PositiveFinite64(policy.MaximumLatencyIncrease) ||
		!checked.Nonzero(policy.MaximumDeviceByteIncrease) {
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
	if !checked.Nonzero(trial.BaselineLatencyNS) || !checked.Nonzero(trial.ComposedLatencyNS) ||
		!checked.Nonzero(trial.BaselinePeakDeviceBytes) || !checked.Nonzero(trial.ComposedPeakDeviceBytes) {
		return errors.New("composition: representation bridge resource evidence is absent")
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

func (policy RepresentationBridgePromotionPolicy) resourceMargins(
	trial RepresentationBridgePromotionTrial,
) (float64, uint64, error) {
	latency := float64(trial.ComposedLatencyNS)/float64(trial.BaselineLatencyNS) - float64(tensor.SingletonExtent)
	var deviceBytes uint64
	if trial.ComposedPeakDeviceBytes > trial.BaselinePeakDeviceBytes {
		deviceBytes = trial.ComposedPeakDeviceBytes - trial.BaselinePeakDeviceBytes
	}
	if !checked.Finite64(latency) {
		return 0, 0, errors.New("composition: representation bridge resource margin is invalid")
	}
	return latency, deviceBytes, nil
}

func (policy RepresentationBridgePromotionPolicy) admit(
	gain, dependence, regression, latency float64,
	deviceBytes uint64,
) error {
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
	if latency > policy.MaximumLatencyIncrease {
		return errors.New("composition: representation bridge exceeds latency allowance")
	}
	if deviceBytes > policy.MaximumDeviceByteIncrease {
		return errors.New("composition: representation bridge exceeds device-memory allowance")
	}
	return nil
}
