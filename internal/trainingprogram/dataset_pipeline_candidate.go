package trainingprogram

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	datasetPipelineCandidateMediaType = "application/vnd.overgo.dataset-pipeline-candidate+json"
	datasetPipelineCandidateSchema    = "overgo/dataset-pipeline-candidate/v1"
)

type datasetOwnerCapability string

const (
	datasetOwnerMaterialization datasetOwnerCapability = "materialization"
	datasetOwnerDeduplication   datasetOwnerCapability = "deduplication"
	datasetOwnerResume          datasetOwnerCapability = "resume"
)

var datasetOwnerCapabilities = [...]datasetOwnerCapability{
	datasetOwnerMaterialization, datasetOwnerDeduplication, datasetOwnerResume,
}

type datasetOwnerFailure struct {
	Capability    datasetOwnerCapability `json:"capability"`
	Owner         artifact.ID            `json:"owner"`
	Requirement   artifact.ID            `json:"requirement"`
	Measurement   artifact.ID            `json:"measurement"`
	Evaluation    artifact.ID            `json:"evaluation"`
	Failed        bool                   `json:"failed"`
	ReopenTrigger artifact.ID            `json:"reopen_trigger"`
}

type datasetPipelineCandidate struct {
	ID        artifact.ID           `json:"-"`
	Version   uint16                `json:"version"`
	Candidate artifact.ID           `json:"candidate"`
	Dataset   artifact.ID           `json:"dataset"`
	Split     artifact.ID           `json:"split"`
	Workload  artifact.ID           `json:"workload"`
	Demand    artifact.ID           `json:"demand"`
	CostBound artifact.ID           `json:"cost_bound"`
	Benefit   artifact.ID           `json:"benefit"`
	Falsifier artifact.ID           `json:"falsifier"`
	Failures  []datasetOwnerFailure `json:"failures"`
}

var datasetPipelineCandidateCodec = artifact.JSONDocumentCodec(
	"dataset pipeline candidate", artifact.KindRecipe,
	datasetPipelineCandidateMediaType, datasetPipelineCandidateSchema,
	canonicalizeDatasetPipelineCandidate,
	func(value datasetPipelineCandidate) artifact.ID { return value.ID },
	func(value *datasetPipelineCandidate, id artifact.ID) { value.ID = id },
	func(value datasetPipelineCandidate) datasetPipelineCandidate {
		value.Failures = slices.Clone(value.Failures)
		return value
	},
)

var datasetPipelineCandidateLineage = func(value datasetPipelineCandidate) []artifact.Lineage {
	parents := []artifact.ID{
		value.Candidate, value.Dataset, value.Split, value.Workload, value.Demand, value.CostBound, value.Benefit, value.Falsifier,
	}
	for _, failure := range value.Failures {
		parents = append(parents,
			failure.Owner, failure.Requirement, failure.Measurement, failure.Evaluation, failure.ReopenTrigger,
		)
	}
	return artifact.DependencyLineage(value.ID, uniqueArtifactIDs(parents)...)
}

func canonicalizeDatasetPipelineCandidate(value *datasetPipelineCandidate) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Candidate.Kind() != artifact.KindRecipe ||
		value.Dataset.Kind() != artifact.KindDataset || value.Split.Kind() != artifact.KindDatasetShard ||
		value.Workload.Kind() != artifact.KindProfile || value.Demand.Kind() != artifact.KindEvidence ||
		value.CostBound.Kind() != artifact.KindEvidence || value.Benefit.Kind() != artifact.KindEvidence ||
		value.Falsifier.Kind() != artifact.KindEvidence || len(value.Failures) != len(datasetOwnerCapabilities) {
		return errors.New("training program: incomplete dataset pipeline demand evidence")
	}
	value.Failures = slices.Clone(value.Failures)
	slices.SortFunc(value.Failures, func(left, right datasetOwnerFailure) int {
		return cmp.Compare(left.Capability, right.Capability)
	})
	required := make(map[datasetOwnerCapability]bool, len(datasetOwnerCapabilities))
	for _, capability := range datasetOwnerCapabilities {
		required[capability] = false
	}
	owners, measurements, evaluations := make(map[artifact.ID]bool), make(map[artifact.ID]bool), make(map[artifact.ID]bool)
	for _, failure := range value.Failures {
		ownerKind := failure.Owner.Kind()
		if _, known := required[failure.Capability]; !known || required[failure.Capability] ||
			(ownerKind != artifact.KindProfile && ownerKind != artifact.KindRecipe) || failure.Owner == value.Candidate || owners[failure.Owner] ||
			failure.Requirement.Kind() != artifact.KindProfile || failure.Measurement.Kind() != artifact.KindEvidence ||
			failure.Evaluation.Kind() != artifact.KindEvidence || failure.Measurement == failure.Evaluation ||
			measurements[failure.Measurement] || evaluations[failure.Evaluation] || !failure.Failed ||
			failure.ReopenTrigger.Kind() != artifact.KindRecipe {
			return errors.New("training program: dataset pipeline candidate lacks a measured existing-owner failure")
		}
		required[failure.Capability], owners[failure.Owner] = true, true
		measurements[failure.Measurement], evaluations[failure.Evaluation] = true, true
	}
	return nil
}
