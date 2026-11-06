package composition_test

import (
	"overgo/internal/composition"
	"overgo/internal/composition/compositiontest"
	"overgo/internal/testcheck"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestCompositeGenerationPromotionGate(t *testing.T) {
	store, authority := compositiontest.Authority(t)
	ctx := t.Context()
	activation := testcheck.Must(authority.Recipe.ActivationBatch(ctx, store, "fixture/composite-generation/promotion-activate", nil))(t)
	if _, err := store.Commit(ctx, activation); err != nil {
		t.Fatal(err)
	}
	plan := testcheck.Must(composition.CompileCompositionExecutionPlan(
		ctx, store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	))(t)
	evidence, dependencies := compositeGenerationPromotionFixture(t, plan)
	promotion := testcheck.Must((composition.CompositeGenerationPromotionAuthority{}).Promote(plan, evidence))(t)
	planContent := testcheck.Must(plan.Content())(t)
	evidenceContent := testcheck.Must(evidence.Content())(t)
	promotionContent := testcheck.Must(promotion.Content())(t)
	alias := testcheck.Must(composition.CompositeGenerationPromotionAlias(plan.SourceModel, plan.TargetModel, plan.Task))(t)
	batch := testcheck.Must(artifact.NewDocumentBatch(
		"fixture/composite-generation/promotion",
		[]artifact.Content{planContent, evidenceContent, promotionContent},
		append(append(plan.Lineage(), evidence.Lineage()...), promotion.Lineage()...),
		[]artifact.AliasBinding{{Name: alias, Target: promotion.ID}},
	))(t)
	batch.Artifacts = append(batch.Artifacts, dependencies...)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	active, activePlan, found, err := composition.ActiveCompositeGeneration(
		ctx, store, plan.SourceModel, plan.TargetModel, plan.Task,
	)
	if err != nil || !found || active.ID != promotion.ID || activePlan.ID != plan.ID {
		t.Fatalf("active promotion = %+v plan=%s found=%t err=%v", active, activePlan.ID, found, err)
	}
	changed := evidence
	changed.ExecutionPlan = testutil.ArtifactID(t, artifact.KindProfile, "foreign execution plan")
	changed.ID = artifact.ID{}
	changed = testcheck.Must((composition.CompositeGenerationCUDAAuthority{}).New(changed))(t)
	if _, err := (composition.CompositeGenerationPromotionAuthority{}).Promote(plan, changed); err == nil {
		t.Fatal("CUDA evidence for a foreign execution plan promoted the composition")
	}
}

func TestCompositeGenerationUnpromotedRefusal(t *testing.T) {
	store, authority := compositiontest.Authority(t)
	ctx := t.Context()
	activation := testcheck.Must(authority.Recipe.ActivationBatch(ctx, store, "fixture/composite-generation/refusal-activate", nil))(t)
	if _, err := store.Commit(ctx, activation); err != nil {
		t.Fatal(err)
	}
	if _, _, found, err := composition.ActiveCompositeGeneration(
		ctx, store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	); err != nil || found {
		t.Fatalf("unpromoted generation = found %t, err %v", found, err)
	}
}

func compositeGenerationPromotionFixture(
	t *testing.T,
	plan composition.CompositionExecutionPlan,
) (composition.CompositeGenerationCUDAEvidence, []artifact.Descriptor) {
	t.Helper()
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	output := id(artifact.KindOutput, "promoted output")
	value := composition.CompositeGenerationCUDAEvidence{
		SourceModel: plan.SourceModel, TargetModel: plan.TargetModel,
		CompositionRecipe: plan.CompositionRecipe, TargetBaselineRecipe: id(artifact.KindRecipe, "native baseline"),
		ExecutionPlan: plan.ID, Promotion: plan.Promotion, Bridge: plan.BridgeWeights[0],
		HeldOutInputs:  []artifact.ID{id(artifact.KindOutput, "held-out input")},
		BaselineOutput: output, ComposedOutput: output,
		BaselineRun: id(artifact.KindRun, "baseline run"), ComposedRun: id(artifact.KindRun, "composed run"),
		BaselineObservation: id(artifact.KindEvidence, "baseline observation"),
		ComposedObservation: id(artifact.KindEvidence, "composed observation"),
		Device:              id(artifact.KindEvidence, "CUDA device"), BridgeExecution: id(artifact.KindEvidence, "CUDA bridge"),
		ExactOutputParity: true,
	}
	evidence := testcheck.Must((composition.CompositeGenerationCUDAAuthority{}).New(value))(t)
	parents := evidence.Lineage()
	descriptors := make([]artifact.Descriptor, 0, len(parents))
	for _, edge := range parents {
		if edge.Parent == plan.SourceModel || edge.Parent == plan.TargetModel ||
			edge.Parent == plan.CompositionRecipe || edge.Parent == plan.ID ||
			edge.Parent == plan.Promotion || edge.Parent == plan.BridgeWeights[0] {
			continue
		}
		descriptors = append(descriptors, artifact.Descriptor{ID: edge.Parent, Size: 1})
	}
	return evidence, descriptors
}
