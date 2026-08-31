package trainingprogram

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestDatasetPipelineCandidateRequiresMeasuredExistingOwnerFailure(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	failure := func(capability datasetOwnerCapability) datasetOwnerFailure {
		name := string(capability)
		return datasetOwnerFailure{
			Capability: capability, Owner: id(artifact.KindProfile, name+" owner"),
			Requirement: id(artifact.KindProfile, name+" requirement"), Measurement: id(artifact.KindEvidence, name+" measurement"),
			Evaluation: id(artifact.KindEvidence, name+" evaluation"), Failed: true,
			ReopenTrigger: id(artifact.KindRecipe, name+" reopen"),
		}
	}
	candidate, err := datasetPipelineCandidateCodec.New(datasetPipelineCandidate{
		Version: artifact.InitialDocumentVersion, Candidate: id(artifact.KindRecipe, "pipeline candidate"),
		Dataset: id(artifact.KindDataset, "dataset"), Split: id(artifact.KindDatasetShard, "split"),
		Workload: id(artifact.KindProfile, "workload"), Demand: id(artifact.KindEvidence, "demand"),
		CostBound: id(artifact.KindEvidence, "cost"), Benefit: id(artifact.KindEvidence, "benefit"),
		Falsifier: id(artifact.KindEvidence, "falsifier"),
		Failures: []datasetOwnerFailure{
			failure(datasetOwnerResume), failure(datasetOwnerMaterialization), failure(datasetOwnerDeduplication),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := datasetPipelineCandidateCodec.Content(candidate)
	if err != nil || content.Descriptor.ID != candidate.ID || len(datasetPipelineCandidateLineage(candidate)) != 23 {
		t.Fatalf("dataset pipeline candidate = (%+v, %v)", candidate, err)
	}
	sufficientOwner := candidate
	sufficientOwner.ID, sufficientOwner.Failures = artifact.ID{}, slices.Clone(candidate.Failures)
	sufficientOwner.Failures[0].Failed = false
	if _, err := datasetPipelineCandidateCodec.New(sufficientOwner); err == nil {
		t.Fatal("candidate admitted while an existing owner remained sufficient")
	}
	hypothetical := candidate
	hypothetical.ID, hypothetical.Failures = artifact.ID{}, slices.Clone(candidate.Failures)
	hypothetical.Failures[1].Measurement = artifact.ID{}
	if _, err := datasetPipelineCandidateCodec.New(hypothetical); err == nil {
		t.Fatal("hypothetical owner failure admitted without measurement")
	}
	duplicateOwner := candidate
	duplicateOwner.ID, duplicateOwner.Failures = artifact.ID{}, slices.Clone(candidate.Failures)
	duplicateOwner.Failures[1].Owner = duplicateOwner.Failures[0].Owner
	if _, err := datasetPipelineCandidateCodec.New(duplicateOwner); err == nil {
		t.Fatal("one failed owner was reused for multiple capabilities")
	}
}
