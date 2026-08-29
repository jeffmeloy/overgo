package evaluation

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/composition"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
	"overgo/internal/textcheck"
)

const (
	// CompositeGenerationQualityVersion is the immutable quality document version.
	CompositeGenerationQualityVersion = artifact.InitialDocumentVersion
	// CompositeGenerationQualityMediaType identifies held-out quality evidence.
	CompositeGenerationQualityMediaType = "application/vnd.overgo.composite-generation-quality+json"
	// CompositeGenerationQualitySchema identifies the held-out quality wire schema.
	CompositeGenerationQualitySchema = "overgo/composite-generation-quality/v1"
	// CompositeGenerationQualityPolicyMediaType identifies recipe-owned admission parameters.
	CompositeGenerationQualityPolicyMediaType = "application/vnd.overgo.composite-generation-quality-policy+json"
	// CompositeGenerationQualityPolicySchema identifies the policy wire schema.
	CompositeGenerationQualityPolicySchema = "overgo/composite-generation-quality-policy/v1"
)

// CompositeGenerationQualityPolicy declares named metrics and the complete
// held-out evidence envelope required for admission.
type CompositeGenerationQualityPolicy struct {
	Version                  uint16              `json:"version"`
	QualityMetric            string              `json:"quality_metric"`
	NondegeneracyMetric      string              `json:"nondegeneracy_metric"`
	Direction                runrecord.Direction `json:"direction"`
	MinimumSeeds             uint64              `json:"minimum_seeds"`
	ComposedQualityThreshold float64             `json:"composed_quality_threshold"`
	MinimumHeldOutGain       float64             `json:"minimum_held_out_gain"`
	MinimumSourceEffect      float64             `json:"minimum_source_effect"`
	MinimumBridgeEffect      float64             `json:"minimum_bridge_effect"`
	MaximumSeedSpread        float64             `json:"maximum_seed_spread"`
	MinimumNondegeneracy     float64             `json:"minimum_nondegeneracy"`
	ID                       artifact.ID         `json:"-"`
}

// CompositeGenerationQuality is derived, immutable admission evidence for one
// exact composite-generation execution matrix.
type CompositeGenerationQuality struct {
	Version              uint16                           `json:"version"`
	GenerationEvidence   artifact.ID                      `json:"generation_evidence"`
	PolicyID             artifact.ID                      `json:"policy_id"`
	Policy               CompositeGenerationQualityPolicy `json:"policy"`
	WorstComposedQuality float64                          `json:"worst_composed_quality"`
	WorstHeldOutGain     float64                          `json:"worst_held_out_gain"`
	WorstSourceEffect    float64                          `json:"worst_source_effect"`
	WorstBridgeEffect    float64                          `json:"worst_bridge_effect"`
	WorstSeedSpread      float64                          `json:"worst_seed_spread"`
	WorstNondegeneracy   float64                          `json:"worst_nondegeneracy"`
	ID                   artifact.ID                      `json:"-"`
}

// CompositeGenerationQualityAuthority owns validation but cannot execute or
// mutate either model.
type CompositeGenerationQualityAuthority struct{}

var compositeGenerationQualityPolicyCodec = artifact.JSONDocumentCodec(
	"composite generation quality policy", artifact.KindProfile,
	CompositeGenerationQualityPolicyMediaType, CompositeGenerationQualityPolicySchema,
	canonicalizeCompositeGenerationQualityPolicy,
	func(value CompositeGenerationQualityPolicy) artifact.ID { return value.ID },
	func(value *CompositeGenerationQualityPolicy, id artifact.ID) { value.ID = id },
	func(value CompositeGenerationQualityPolicy) CompositeGenerationQualityPolicy { return value },
)

var compositeGenerationQualityCodec = artifact.JSONDocumentCodec(
	"composite generation quality", artifact.KindEvidence,
	CompositeGenerationQualityMediaType, CompositeGenerationQualitySchema,
	canonicalizeCompositeGenerationQuality,
	func(value CompositeGenerationQuality) artifact.ID { return value.ID },
	func(value *CompositeGenerationQuality, id artifact.ID) { value.ID = id },
	func(value CompositeGenerationQuality) CompositeGenerationQuality { return value },
)

// NewPolicy validates and identifies recipe-owned quality parameters.
func (CompositeGenerationQualityAuthority) NewPolicy(
	value CompositeGenerationQualityPolicy,
) (CompositeGenerationQualityPolicy, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return compositeGenerationQualityPolicyCodec.New(value)
}

// Evaluate loads exact execution records and derives worst-case quality evidence.
func (CompositeGenerationQualityAuthority) Evaluate(
	ctx context.Context,
	reader artifact.Reader,
	generationEvidence artifact.ID,
	policy CompositeGenerationQualityPolicy,
) (CompositeGenerationQuality, error) {
	if ctx == nil || reader == nil || compositeGenerationQualityPolicyCodec.ValidateIdentity(policy) != nil {
		return CompositeGenerationQuality{}, errors.New("evaluation: composite generation quality authority is absent")
	}
	evidence, err := (composition.CompositeGenerationEvidenceAuthority{}).Load(ctx, reader, generationEvidence)
	if err != nil {
		return CompositeGenerationQuality{}, err
	}
	type caseSeed struct {
		Input artifact.ID
		Seed  uint64
	}
	type scores struct {
		quality       map[composition.CompositeGenerationArm]float64
		nondegeneracy map[composition.CompositeGenerationArm]float64
	}
	groups := make(map[caseSeed]scores)
	inputSeeds := make(map[artifact.ID]map[uint64]struct{})
	for _, trial := range evidence.Trials {
		quality, nondegeneracy, err := loadCompositeGenerationTrialScores(ctx, reader, evidence, policy, trial)
		if err != nil {
			return CompositeGenerationQuality{}, err
		}
		key := caseSeed{Input: trial.Input, Seed: trial.Seed}
		group := groups[key]
		if group.quality == nil {
			group.quality = make(map[composition.CompositeGenerationArm]float64)
			group.nondegeneracy = make(map[composition.CompositeGenerationArm]float64)
		}
		group.quality[trial.Arm], group.nondegeneracy[trial.Arm] = quality, nondegeneracy
		groups[key] = group
		if inputSeeds[trial.Input] == nil {
			inputSeeds[trial.Input] = make(map[uint64]struct{})
		}
		inputSeeds[trial.Input][trial.Seed] = struct{}{}
	}
	for _, seeds := range inputSeeds {
		if uint64(len(seeds)) < policy.MinimumSeeds {
			return CompositeGenerationQuality{}, errors.New("evaluation: composite generation lacks repeated held-out seeds")
		}
	}
	first := true
	var result CompositeGenerationQuality
	composedByInput := make(map[artifact.ID][]float64)
	for key, group := range groups {
		composed := group.quality[composition.CompositeGenerationComposed]
		gain, _ := policy.Direction.Advantage(composed, group.quality[composition.CompositeGenerationTargetBaseline])
		sourceEffect, _ := policy.Direction.Advantage(composed, group.quality[composition.CompositeGenerationSourceAblated])
		bridgeEffect, _ := policy.Direction.Advantage(composed, group.quality[composition.CompositeGenerationBridgeAblated])
		nondegeneracy := group.nondegeneracy[composition.CompositeGenerationComposed]
		if !policy.Direction.Admits(composed, policy.ComposedQualityThreshold) || gain < policy.MinimumHeldOutGain ||
			sourceEffect < policy.MinimumSourceEffect || bridgeEffect < policy.MinimumBridgeEffect ||
			nondegeneracy < policy.MinimumNondegeneracy {
			return CompositeGenerationQuality{}, errors.New("evaluation: composite generation quality envelope was not met")
		}
		if first {
			result.WorstComposedQuality, result.WorstHeldOutGain = composed, gain
			result.WorstSourceEffect, result.WorstBridgeEffect = sourceEffect, bridgeEffect
			result.WorstNondegeneracy, first = nondegeneracy, false
		} else {
			result.WorstComposedQuality = policy.Direction.Worse(result.WorstComposedQuality, composed)
			result.WorstHeldOutGain = min(result.WorstHeldOutGain, gain)
			result.WorstSourceEffect = min(result.WorstSourceEffect, sourceEffect)
			result.WorstBridgeEffect = min(result.WorstBridgeEffect, bridgeEffect)
			result.WorstNondegeneracy = min(result.WorstNondegeneracy, nondegeneracy)
		}
		composedByInput[key.Input] = append(composedByInput[key.Input], composed)
	}
	for _, values := range composedByInput {
		slices.Sort(values)
		spread := values[len(values)-tensor.SingletonExtent] - values[tensor.FirstOffset]
		if spread > policy.MaximumSeedSpread {
			return CompositeGenerationQuality{}, errors.New("evaluation: composite generation repeated-seed stability is insufficient")
		}
		result.WorstSeedSpread = max(result.WorstSeedSpread, spread)
	}
	result.Version, result.GenerationEvidence = CompositeGenerationQualityVersion, evidence.ID
	result.PolicyID, result.Policy = policy.ID, policy
	return compositeGenerationQualityCodec.New(result)
}

// Parse admits canonical serialized quality evidence.
func (CompositeGenerationQualityAuthority) Parse(data []byte) (CompositeGenerationQuality, error) {
	return compositeGenerationQualityCodec.Parse(data)
}

// ValidateIdentity verifies derived fields and exact content identity.
func (value CompositeGenerationQuality) ValidateIdentity() error {
	return compositeGenerationQualityCodec.ValidateIdentity(value)
}

// Content returns exact quality evidence content.
func (value CompositeGenerationQuality) Content() (artifact.Content, error) {
	return compositeGenerationQualityCodec.Content(value)
}

// Lineage binds quality admission to the execution evidence and recipe policy.
func (value CompositeGenerationQuality) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.GenerationEvidence, value.PolicyID)
}

// Batch prepares atomic quality-evidence publication.
func (value CompositeGenerationQuality) Batch(key string) (artifact.Batch, error) {
	return compositeGenerationQualityCodec.Batch(key, value, value.Lineage(), nil)
}

// Content returns exact policy content.
func (policy CompositeGenerationQualityPolicy) Content() (artifact.Content, error) {
	return compositeGenerationQualityPolicyCodec.Content(policy)
}

func loadCompositeGenerationTrialScores(
	ctx context.Context,
	reader artifact.Reader,
	evidence composition.CompositeGenerationEvidence,
	policy CompositeGenerationQualityPolicy,
	trial composition.CompositeGenerationTrial,
) (float64, float64, error) {
	content, found, err := artifact.ReadContent(ctx, reader, trial.Output)
	if err != nil || !found || content.Descriptor.ID != trial.Output || content.Validate() != nil {
		return 0, 0, errors.Join(err, errors.New("evaluation: composite generation output content is absent"))
	}
	run, err := runrecord.RequireRun(ctx, reader, trial.Run)
	if err != nil {
		return 0, 0, err
	}
	evaluation, err := runrecord.RequireEvaluation(ctx, reader, trial.Evaluation)
	if err != nil {
		return 0, 0, err
	}
	wantRecipe := compositeGenerationTrialRecipe(evidence, trial.Arm)
	if run.Outcome != runrecord.OutcomeSucceeded || run.Recipe != wantRecipe ||
		!slices.Contains(run.Inputs, trial.Input) || !slices.Contains(run.Outputs, trial.Output) ||
		evaluation.Recipe != wantRecipe || evaluation.Run != run.ID || evaluation.Dataset != evidence.Dataset {
		return 0, 0, errors.New("evaluation: composite generation run lineage differs")
	}
	metrics := make(map[string]float64, len(evaluation.Metrics))
	for _, metric := range evaluation.Metrics {
		metrics[metric.Name] = metric.Value
	}
	quality, qualityFound := metrics[policy.QualityMetric]
	nondegeneracy, nondegeneracyFound := metrics[policy.NondegeneracyMetric]
	if !qualityFound || !nondegeneracyFound {
		return 0, 0, errors.New("evaluation: composite generation required metric is absent")
	}
	return quality, nondegeneracy, nil
}

func compositeGenerationTrialRecipe(
	evidence composition.CompositeGenerationEvidence,
	arm composition.CompositeGenerationArm,
) artifact.ID {
	switch arm {
	case composition.CompositeGenerationTargetBaseline:
		return evidence.TargetBaselineRecipe
	case composition.CompositeGenerationComposed:
		return evidence.CompositionRecipe
	case composition.CompositeGenerationSourceAblated:
		return evidence.SourceAblatedRecipe
	case composition.CompositeGenerationBridgeAblated:
		return evidence.BridgeAblatedRecipe
	default:
		return artifact.ID{}
	}
}

func canonicalizeCompositeGenerationQualityPolicy(value *CompositeGenerationQualityPolicy) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		!textcheck.LowerIdentifier(value.QualityMetric, len(value.QualityMetric)) ||
		!textcheck.LowerIdentifier(value.NondegeneracyMetric, len(value.NondegeneracyMetric)) ||
		value.QualityMetric == value.NondegeneracyMetric ||
		value.Direction != runrecord.DirectionMinimize && value.Direction != runrecord.DirectionMaximize ||
		value.MinimumSeeds < uint64(tensor.PairedExtent) ||
		!checked.Finite64(value.ComposedQualityThreshold) ||
		!checked.PositiveFinite64(value.MinimumHeldOutGain) ||
		!checked.PositiveFinite64(value.MinimumSourceEffect) ||
		!checked.PositiveFinite64(value.MinimumBridgeEffect) ||
		!checked.PositiveFinite64(value.MaximumSeedSpread) ||
		!checked.PositiveFinite64(value.MinimumNondegeneracy) {
		return errors.New("evaluation: invalid composite generation quality policy")
	}
	return nil
}

func canonicalizeCompositeGenerationQuality(value *CompositeGenerationQuality) error {
	if value == nil || value.Version != CompositeGenerationQualityVersion ||
		value.GenerationEvidence.Kind() != artifact.KindEvidence || value.PolicyID.Kind() != artifact.KindProfile {
		return errors.New("evaluation: invalid composite generation quality evidence")
	}
	policy, err := (CompositeGenerationQualityAuthority{}).NewPolicy(value.Policy)
	if err != nil || policy.ID != value.PolicyID {
		return errors.Join(err, errors.New("evaluation: composite generation quality policy differs"))
	}
	value.Policy = policy
	for _, metric := range []float64{
		value.WorstComposedQuality, value.WorstHeldOutGain, value.WorstSourceEffect,
		value.WorstBridgeEffect, value.WorstSeedSpread, value.WorstNondegeneracy,
	} {
		if !checked.Finite64(metric) {
			return errors.New("evaluation: composite generation quality result is non-finite")
		}
	}
	if !policy.Direction.Admits(value.WorstComposedQuality, policy.ComposedQualityThreshold) ||
		value.WorstHeldOutGain < policy.MinimumHeldOutGain ||
		value.WorstSourceEffect < policy.MinimumSourceEffect ||
		value.WorstBridgeEffect < policy.MinimumBridgeEffect ||
		value.WorstSeedSpread > policy.MaximumSeedSpread ||
		value.WorstNondegeneracy < policy.MinimumNondegeneracy {
		return errors.New("evaluation: composite generation quality result does not satisfy policy")
	}
	return nil
}
