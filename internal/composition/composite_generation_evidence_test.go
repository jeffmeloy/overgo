package composition

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestCompositeGenerationEvidence(t *testing.T) {
	value := compositeGenerationEvidenceFixture(t)
	authority := CompositeGenerationEvidenceAuthority{}
	evidence, err := authority.New(value)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.ID.Kind() != artifact.KindEvidence || evidence.Trials[0].Seed != 7 ||
		evidence.Trials[0].Arm != CompositeGenerationTargetBaseline {
		t.Fatalf("evidence = %+v", evidence)
	}
	content, err := evidence.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := authority.Parse(content.Data)
	if err != nil || parsed.ID != evidence.ID || !slices.Equal(parsed.Trials, evidence.Trials) {
		t.Fatalf("round trip = %+v, %v", parsed, err)
	}
	if len(evidence.Lineage()) != 28 {
		t.Fatalf("lineage count = %d", len(evidence.Lineage()))
	}
}

func TestCompositeGenerationFrozenModels(t *testing.T) {
	value := compositeGenerationEvidenceFixture(t)
	value.SourceModel.After.Size++
	if _, err := (CompositeGenerationEvidenceAuthority{}).New(value); err == nil {
		t.Fatal("changed source descriptor admitted")
	}
	value = compositeGenerationEvidenceFixture(t)
	value.TargetModel.After.ID = testutil.ArtifactID(t, artifact.KindModel, "changed target")
	if _, err := (CompositeGenerationEvidenceAuthority{}).New(value); err == nil {
		t.Fatal("changed target identity admitted")
	}
}

func TestCompositeGenerationAblationArms(t *testing.T) {
	value := compositeGenerationEvidenceFixture(t)
	value.Trials = value.Trials[:len(value.Trials)-1]
	if _, err := (CompositeGenerationEvidenceAuthority{}).New(value); err == nil {
		t.Fatal("incomplete four-arm matrix admitted")
	}
	value = compositeGenerationEvidenceFixture(t)
	value.Trials[1].Arm = CompositeGenerationTargetBaseline
	if _, err := (CompositeGenerationEvidenceAuthority{}).New(value); err == nil {
		t.Fatal("duplicate arm admitted")
	}
	value = compositeGenerationEvidenceFixture(t)
	value.Trials[1].Observation = value.Trials[0].Observation
	if _, err := (CompositeGenerationEvidenceAuthority{}).New(value); err == nil {
		t.Fatal("reused resource observation admitted")
	}
}

func TestCompositeGenerationIdentity(t *testing.T) {
	value := compositeGenerationEvidenceFixture(t)
	authority := CompositeGenerationEvidenceAuthority{}
	first, err := authority.New(value)
	if err != nil {
		t.Fatal(err)
	}
	slices.Reverse(value.Trials)
	second, err := authority.New(value)
	if err != nil || first.ID != second.ID {
		t.Fatalf("canonical identities = %s and %s: %v", first.ID, second.ID, err)
	}
	second.Trials[0].Output = testutil.ArtifactID(t, artifact.KindOutput, "replacement output")
	if second.ValidateIdentity() == nil {
		t.Fatal("identity drift accepted")
	}
}

func compositeGenerationEvidenceFixture(t *testing.T) CompositeGenerationEvidence {
	t.Helper()
	id := func(kind artifact.Kind, label string) artifact.ID {
		return testutil.ArtifactID(t, kind, label)
	}
	source := artifact.Descriptor{ID: id(artifact.KindModel, "source model"), Size: 101, MediaType: "application/octet-stream"}
	target := artifact.Descriptor{ID: id(artifact.KindModel, "target model"), Size: 103, MediaType: "application/octet-stream"}
	value := CompositeGenerationEvidence{
		SourceModel:          CompositeGenerationFrozenModel{Before: source, After: source},
		TargetModel:          CompositeGenerationFrozenModel{Before: target, After: target},
		Bridge:               artifact.Descriptor{ID: id(artifact.KindAdapter, "bridge weights"), Size: 107, MediaType: "application/octet-stream"},
		ExecutionPlan:        id(artifact.KindProfile, "composition execution plan"),
		TargetBaselineRecipe: id(artifact.KindRecipe, "target baseline recipe"),
		CompositionRecipe:    id(artifact.KindRecipe, "active composition recipe"),
		SourceAblatedRecipe:  id(artifact.KindRecipe, "source ablated recipe"),
		BridgeAblatedRecipe:  id(artifact.KindRecipe, "bridge ablated recipe"),
		Dataset:              id(artifact.KindDataset, "heldout dataset"),
		HeldOutSplit:         id(artifact.KindDatasetShard, "heldout split"),
		Evaluator:            id(artifact.KindEvidence, "generation evaluator"),
	}
	for _, arm := range compositeGenerationArms {
		value.Trials = append(value.Trials, CompositeGenerationTrial{
			Seed: 7, Arm: arm,
			Input:       id(artifact.KindOutput, "heldout input"),
			Output:      id(artifact.KindOutput, "generated output "+arm.String()),
			Run:         id(artifact.KindRun, "generation run "+arm.String()),
			Evaluation:  id(artifact.KindEvaluation, "generation evaluation "+arm.String()),
			Observation: id(artifact.KindEvidence, "generation observation "+arm.String()),
		})
	}
	return value
}

func (arm CompositeGenerationArm) String() string { return string(arm) }
