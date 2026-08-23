package composition

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/safetensors"
)

const (
	// OfflineArtifactGenerationEvidenceVersion is the immutable evidence version.
	OfflineArtifactGenerationEvidenceVersion = artifact.InitialDocumentVersion
	// OfflineArtifactGenerationEvidenceMediaType identifies load and generation evidence.
	OfflineArtifactGenerationEvidenceMediaType = "application/vnd.overgo.offline-artifact-generation-evidence+json"
	// OfflineArtifactGenerationEvidenceSchema identifies the evidence wire schema.
	OfflineArtifactGenerationEvidenceSchema = "overgo/offline-artifact-generation-evidence/v1"
)

// OfflineArtifactGenerationTrial binds one seeded generation to its output,
// run, serving observation, latency, and peak host-memory facts.
type OfflineArtifactGenerationTrial struct {
	Seed          uint64      `json:"seed"`
	Output        artifact.ID `json:"output"`
	Run           artifact.ID `json:"run"`
	Observation   artifact.ID `json:"observation"`
	LatencyNS     uint64      `json:"latency_ns"`
	PeakHostBytes uint64      `json:"peak_host_bytes"`
}

// OfflineArtifactGenerationEvidence records a real load of the produced
// Safetensors catalog together with generation trials from that exact model.
type OfflineArtifactGenerationEvidence struct {
	Version       uint16                           `json:"version"`
	ExecutionPlan artifact.ID                      `json:"execution_plan"`
	ProducedModel artifact.ID                      `json:"produced_model"`
	Operator      OfflineArtifactOperator          `json:"operator"`
	TensorCount   uint32                           `json:"tensor_count"`
	TensorBytes   uint64                           `json:"tensor_bytes"`
	Trials        []OfflineArtifactGenerationTrial `json:"trials"`
	ID            artifact.ID                      `json:"-"`
}

var offlineArtifactGenerationEvidenceCodec = artifact.JSONDocumentCodec(
	"offline artifact generation evidence", artifact.KindEvidence,
	OfflineArtifactGenerationEvidenceMediaType, OfflineArtifactGenerationEvidenceSchema,
	canonicalizeOfflineArtifactGenerationEvidence,
	func(value OfflineArtifactGenerationEvidence) artifact.ID { return value.ID },
	func(value *OfflineArtifactGenerationEvidence, id artifact.ID) { value.ID = id },
	func(value OfflineArtifactGenerationEvidence) OfflineArtifactGenerationEvidence {
		value.Trials = slices.Clone(value.Trials)
		return value
	},
)

// ValidateOfflineArtifactGeneration opens the produced directory, checks its
// complete catalog against the execution plan, and seals supplied generation
// run facts as immutable evidence.
func ValidateOfflineArtifactGeneration(
	plan OfflineTensorExecutionPlan,
	directory string,
	producedModel artifact.ID,
	trials []OfflineArtifactGenerationTrial,
) (OfflineArtifactGenerationEvidence, error) {
	if err := plan.ValidateIdentity(); err != nil || producedModel.Kind() != artifact.KindModel {
		return OfflineArtifactGenerationEvidence{}, errors.Join(
			errors.New("composition: offline generated artifact authority differs"), err,
		)
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return OfflineArtifactGenerationEvidence{}, fmt.Errorf("composition: load offline generated artifact: %w", err)
	}
	defer source.Close()
	if len(source.Tensors) != len(plan.Operations) || len(source.Shards()) != len(plan.Shards) {
		return OfflineArtifactGenerationEvidence{}, errors.New("composition: generated artifact catalog differs from execution plan")
	}
	var total uint64
	for _, operation := range plan.Operations {
		loaded, found := source.Tensors[operation.Name]
		if !found || !strings.EqualFold(loaded.DType, operation.Storage) || uint64(loaded.Size()) != operation.OutputBytes {
			return OfflineArtifactGenerationEvidence{}, fmt.Errorf("composition: generated tensor %q differs from execution plan", operation.Name)
		}
		var ok bool
		total, ok = checked.Add64(total, operation.OutputBytes)
		if !ok {
			return OfflineArtifactGenerationEvidence{}, errors.New("composition: generated tensor extent overflows")
		}
	}
	return offlineArtifactGenerationEvidenceCodec.New(OfflineArtifactGenerationEvidence{
		Version:       OfflineArtifactGenerationEvidenceVersion,
		ExecutionPlan: plan.ID, ProducedModel: producedModel, Operator: plan.Operator,
		TensorCount: uint32(len(plan.Operations)), TensorBytes: total, Trials: trials,
	})
}

// ValidateIdentity verifies the generation evidence content identity.
func (value OfflineArtifactGenerationEvidence) ValidateIdentity() error {
	return offlineArtifactGenerationEvidenceCodec.ValidateIdentity(value)
}

// Content returns exact generation evidence content.
func (value OfflineArtifactGenerationEvidence) Content() (artifact.Content, error) {
	return offlineArtifactGenerationEvidenceCodec.Content(value)
}

// Lineage binds generation evidence to its plan, model, outputs, runs, and observations.
func (value OfflineArtifactGenerationEvidence) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.ExecutionPlan, value.ProducedModel}
	for _, trial := range value.Trials {
		parents = append(parents, trial.Output, trial.Run, trial.Observation)
	}
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

// Batch prepares atomic publication of generation evidence and lineage.
func (value OfflineArtifactGenerationEvidence) Batch(key string) (artifact.Batch, error) {
	return offlineArtifactGenerationEvidenceCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeOfflineArtifactGenerationEvidence(value *OfflineArtifactGenerationEvidence) error {
	if value == nil || value.Version != OfflineArtifactGenerationEvidenceVersion ||
		value.ExecutionPlan.Kind() != artifact.KindProfile || value.ProducedModel.Kind() != artifact.KindModel ||
		(value.Operator != OfflineArtifactExactPassthrough && value.Operator != OfflineArtifactTaskArithmetic) ||
		!checked.Nonzero(value.TensorCount) || !checked.Nonzero(value.TensorBytes) || !checked.Nonempty(value.Trials) {
		return errors.New("composition: invalid offline artifact generation evidence")
	}
	for _, trial := range value.Trials {
		if trial.Output.Kind() != artifact.KindOutput || trial.Run.Kind() != artifact.KindRun ||
			trial.Observation.Kind() != artifact.KindEvidence || !checked.Nonzero(trial.LatencyNS) ||
			!checked.Nonzero(trial.PeakHostBytes) {
			return errors.New("composition: invalid offline artifact generation trial")
		}
	}
	slices.SortFunc(value.Trials, func(left, right OfflineArtifactGenerationTrial) int {
		if order := cmp.Compare(left.Seed, right.Seed); order != 0 {
			return order
		}
		return cmp.Compare(left.Run.String(), right.Run.String())
	})
	seenRuns := make(map[artifact.ID]bool, len(value.Trials))
	seenObservations := make(map[artifact.ID]bool, len(value.Trials))
	for _, trial := range value.Trials {
		if seenRuns[trial.Run] || seenObservations[trial.Observation] {
			return errors.New("composition: offline generation run is duplicated")
		}
		seenRuns[trial.Run], seenObservations[trial.Observation] = true, true
	}
	return nil
}
