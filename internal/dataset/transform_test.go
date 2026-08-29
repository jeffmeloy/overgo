package dataset

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestDatasetTransformIdentityRejectsCallableAndAdvisoryVersionSemantics(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	value, err := transformCodec.New(transform{
		Version: artifact.InitialDocumentVersion, IdentityMode: transformContentAddressed,
		Registration: id(artifact.KindProfile, "registration"), Operation: id(artifact.KindRecipe, "operation"),
		Parameters: id(artifact.KindRecipe, "parameters"), Code: id(artifact.KindEvidence, "code"),
		Environment: id(artifact.KindEvidence, "environment"), Inputs: []artifact.ID{id(artifact.KindDataset, "input")},
		Outputs: []artifact.ID{id(artifact.KindDatasetShard, "output")}, ReplayEvidence: id(artifact.KindEvidence, "replay"),
		Deterministic: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := transformCodec.Content(value)
	if err != nil || content.Descriptor.ID != value.ID || len(transformLineage(value)) != 8 {
		t.Fatalf("dataset transform = (%+v, %v)", value, err)
	}
	callable := value
	callable.ID, callable.Callable = artifact.ID{}, "package.function"
	if _, err := transformCodec.New(callable); err == nil {
		t.Fatal("callable identity admitted")
	}
	advisoryVersion := value
	advisoryVersion.ID, advisoryVersion.AdvisoryVersion = artifact.ID{}, "latest"
	if _, err := transformCodec.New(advisoryVersion); err == nil {
		t.Fatal("advisory name-at-version identity admitted")
	}
	identityOutput := value
	identityOutput.ID, identityOutput.Outputs = artifact.ID{}, slices.Clone(value.Inputs)
	if _, err := transformCodec.New(identityOutput); err == nil {
		t.Fatal("output reused an input identity")
	}
}
