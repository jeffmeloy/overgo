package composition

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestCompositeGenerationCUDAEvidence(t *testing.T) {
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	output := id(artifact.KindOutput, "identical CUDA output")
	value := CompositeGenerationCUDAEvidence{
		SourceModel: id(artifact.KindModel, "CUDA source"), TargetModel: id(artifact.KindModel, "CUDA target"),
		CompositionRecipe: id(artifact.KindRecipe, "CUDA composition"), TargetBaselineRecipe: id(artifact.KindRecipe, "CUDA baseline"),
		ExecutionPlan: id(artifact.KindProfile, "CUDA execution plan"), Promotion: id(artifact.KindEvidence, "CUDA promotion"),
		Bridge: id(artifact.KindAdapter, "CUDA bridge"), HeldOutInputs: []artifact.ID{id(artifact.KindOutput, "heldout input")},
		BaselineOutput: output, ComposedOutput: output,
		BaselineRun: id(artifact.KindRun, "CUDA baseline run"), ComposedRun: id(artifact.KindRun, "CUDA composed run"),
		BaselineObservation: id(artifact.KindEvidence, "CUDA baseline observation"),
		ComposedObservation: id(artifact.KindEvidence, "CUDA composed observation"),
		Device:              id(artifact.KindEvidence, "CUDA device"), BridgeExecution: id(artifact.KindEvidence, "CUDA bridge execution"),
		ExactOutputParity: true,
	}
	evidence, err := (CompositeGenerationCUDAAuthority{}).New(value)
	if err != nil {
		t.Fatal(err)
	}
	content, err := evidence.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := (CompositeGenerationCUDAAuthority{}).Parse(content.Data)
	if err != nil || parsed.ID != evidence.ID || parsed.ValidateIdentity() != nil {
		t.Fatalf("CUDA evidence round trip = %+v, %v", parsed, err)
	}
	value.ExactOutputParity = false
	if _, err := (CompositeGenerationCUDAAuthority{}).New(value); err == nil {
		t.Fatal("non-parity CUDA evidence admitted")
	}
}
