package evaluation

import (
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/composition"
)

const (
	// offlineArtifactGenerationPolicyVersion is the immutable policy version.
	offlineArtifactGenerationPolicyVersion = artifact.InitialDocumentVersion
	// offlineArtifactGenerationPolicyMediaType identifies generation gate policy.
	offlineArtifactGenerationPolicyMediaType = "application/vnd.overgo.offline-artifact-generation-policy+json"
	// offlineArtifactGenerationPolicySchema identifies the policy wire schema.
	offlineArtifactGenerationPolicySchema = "overgo/offline-artifact-generation-policy/v1"
	// offlineArtifactGenerationPromotionVersion is the promotion-evidence version.
	offlineArtifactGenerationPromotionVersion = artifact.InitialDocumentVersion
	// offlineArtifactGenerationPromotionMediaType identifies accepted generation evidence.
	offlineArtifactGenerationPromotionMediaType = "application/vnd.overgo.offline-artifact-generation-promotion+json"
	// offlineArtifactGenerationPromotionSchema identifies promotion evidence.
	offlineArtifactGenerationPromotionSchema = "overgo/offline-artifact-generation-promotion/v1"
)

// offlineArtifactGenerationPolicy owns every numeric generation gate.
type offlineArtifactGenerationPolicy struct {
	Version              uint16      `json:"version"`
	MinimumDistinctSeeds uint32      `json:"minimum_distinct_seeds"`
	RunsPerSeed          uint32      `json:"runs_per_seed"`
	MaximumLatencyNS     uint64      `json:"maximum_latency_ns"`
	MaximumPeakHostBytes uint64      `json:"maximum_peak_host_bytes"`
	Rationale            string      `json:"rationale"`
	ReopenTrigger        string      `json:"reopen_trigger"`
	ID                   artifact.ID `json:"-"`
}

// OfflineArtifactGenerationPromotion is accepted load, determinism, and
// resource evidence for one exact produced model and execution plan.
type OfflineArtifactGenerationPromotion struct {
	Version            uint16      `json:"version"`
	ExecutionPlan      artifact.ID `json:"execution_plan"`
	ProducedModel      artifact.ID `json:"produced_model"`
	GenerationEvidence artifact.ID `json:"generation_evidence"`
	Policy             artifact.ID `json:"policy"`
	DistinctSeeds      uint32      `json:"distinct_seeds"`
	RunCount           uint32      `json:"run_count"`
	WorstLatencyNS     uint64      `json:"worst_latency_ns"`
	WorstPeakHostBytes uint64      `json:"worst_peak_host_bytes"`
	ID                 artifact.ID `json:"-"`
}

var offlineArtifactGenerationPolicyCodec = artifact.JSONDocumentCodec(
	"offline artifact generation policy", artifact.KindProfile,
	offlineArtifactGenerationPolicyMediaType, offlineArtifactGenerationPolicySchema,
	canonicalizeOfflineArtifactGenerationPolicy,
	func(value offlineArtifactGenerationPolicy) artifact.ID { return value.ID },
	func(value *offlineArtifactGenerationPolicy, id artifact.ID) { value.ID = id },
)

var offlineArtifactGenerationPromotionCodec = artifact.JSONDocumentCodec(
	"offline artifact generation promotion", artifact.KindEvidence,
	offlineArtifactGenerationPromotionMediaType, offlineArtifactGenerationPromotionSchema,
	canonicalizeOfflineArtifactGenerationPromotion,
	func(value OfflineArtifactGenerationPromotion) artifact.ID { return value.ID },
	func(value *OfflineArtifactGenerationPromotion, id artifact.ID) { value.ID = id },
)

// NewOfflineArtifactGenerationPolicy records explicit acceptance thresholds.
func NewOfflineArtifactGenerationPolicy(
	minimumDistinctSeeds, runsPerSeed uint32,
	maximumLatencyNS, maximumPeakHostBytes uint64,
	rationale, reopenTrigger string,
) (offlineArtifactGenerationPolicy, error) {
	return offlineArtifactGenerationPolicyCodec.New(offlineArtifactGenerationPolicy{
		Version:              offlineArtifactGenerationPolicyVersion,
		MinimumDistinctSeeds: minimumDistinctSeeds, RunsPerSeed: runsPerSeed,
		MaximumLatencyNS: maximumLatencyNS, MaximumPeakHostBytes: maximumPeakHostBytes,
		Rationale: rationale, ReopenTrigger: reopenTrigger,
	})
}

// PromoteOfflineArtifactGeneration admits only deterministic repeated-seed
// generations that remain inside the complete recipe-owned resource policy.
func PromoteOfflineArtifactGeneration(
	evidence composition.OfflineArtifactGenerationEvidence,
	policy offlineArtifactGenerationPolicy,
) (OfflineArtifactGenerationPromotion, error) {
	if evidence.ValidateIdentity() != nil || offlineArtifactGenerationPolicyCodec.ValidateIdentity(policy) != nil {
		return OfflineArtifactGenerationPromotion{}, errors.New("evaluation: offline artifact generation authority differs")
	}
	type seedFact struct {
		output artifact.ID
		runs   uint32
	}
	seeds := make(map[uint64]seedFact)
	var worstLatency, worstHost uint64
	for _, trial := range evidence.Trials {
		fact, found := seeds[trial.Seed]
		if found && fact.output != trial.Output {
			return OfflineArtifactGenerationPromotion{}, errors.New("evaluation: repeated seed output is unstable")
		}
		fact.output = trial.Output
		fact.runs++
		seeds[trial.Seed] = fact
		worstLatency, worstHost = max(worstLatency, trial.LatencyNS), max(worstHost, trial.PeakHostBytes)
		if trial.LatencyNS > policy.MaximumLatencyNS || trial.PeakHostBytes > policy.MaximumPeakHostBytes {
			return OfflineArtifactGenerationPromotion{}, errors.New("evaluation: offline generation exceeds resource policy")
		}
	}
	if uint32(len(seeds)) < policy.MinimumDistinctSeeds {
		return OfflineArtifactGenerationPromotion{}, errors.New("evaluation: offline generation seed coverage is insufficient")
	}
	for _, fact := range seeds {
		if fact.runs < policy.RunsPerSeed {
			return OfflineArtifactGenerationPromotion{}, errors.New("evaluation: offline generation repeat coverage is insufficient")
		}
	}
	return offlineArtifactGenerationPromotionCodec.New(OfflineArtifactGenerationPromotion{
		Version:       offlineArtifactGenerationPromotionVersion,
		ExecutionPlan: evidence.ExecutionPlan, ProducedModel: evidence.ProducedModel,
		GenerationEvidence: evidence.ID, Policy: policy.ID,
		DistinctSeeds: uint32(len(seeds)), RunCount: uint32(len(evidence.Trials)),
		WorstLatencyNS: worstLatency, WorstPeakHostBytes: worstHost,
	})
}

// Content returns exact policy content.
func (value offlineArtifactGenerationPolicy) Content() (artifact.Content, error) {
	return offlineArtifactGenerationPolicyCodec.Content(value)
}

// Content returns exact promotion evidence content.
func (value OfflineArtifactGenerationPromotion) Content() (artifact.Content, error) {
	return offlineArtifactGenerationPromotionCodec.Content(value)
}

// Lineage binds promotion to the exact plan, model, generation evidence, and policy.
func (value OfflineArtifactGenerationPromotion) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(
		value.ID, value.ExecutionPlan, value.ProducedModel, value.GenerationEvidence, value.Policy,
	)
}

func canonicalizeOfflineArtifactGenerationPolicy(value *offlineArtifactGenerationPolicy) error {
	if value == nil || value.Version != offlineArtifactGenerationPolicyVersion ||
		!checked.Nonzero(value.MinimumDistinctSeeds) || !checked.Nonzero(value.RunsPerSeed) ||
		!checked.Nonzero(value.MaximumLatencyNS) || !checked.Nonzero(value.MaximumPeakHostBytes) ||
		strings.TrimSpace(value.Rationale) != value.Rationale || strings.TrimSpace(value.ReopenTrigger) != value.ReopenTrigger ||
		value.Rationale == "" || value.ReopenTrigger == "" ||
		strings.ContainsAny(value.Rationale, "\x00\r\n") || strings.ContainsAny(value.ReopenTrigger, "\x00\r\n") {
		return errors.New("evaluation: invalid offline artifact generation policy")
	}
	return nil
}

func canonicalizeOfflineArtifactGenerationPromotion(value *OfflineArtifactGenerationPromotion) error {
	if value == nil || value.Version != offlineArtifactGenerationPromotionVersion ||
		value.ExecutionPlan.Kind() != artifact.KindProfile || value.ProducedModel.Kind() != artifact.KindModel ||
		value.GenerationEvidence.Kind() != artifact.KindEvidence || value.Policy.Kind() != artifact.KindProfile ||
		!checked.Nonzero(value.DistinctSeeds) || !checked.Nonzero(value.RunCount) ||
		!checked.Nonzero(value.WorstLatencyNS) || !checked.Nonzero(value.WorstPeakHostBytes) {
		return errors.New("evaluation: invalid offline artifact generation promotion")
	}
	return nil
}
