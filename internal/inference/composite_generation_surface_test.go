package inference

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/overgodb"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestCompositeGenerationSurfaceParity(t *testing.T) {
	store, authority := productionCompositionFixture(t, true)
	runtime, err := OpenProductionComposition(
		context.Background(), store, authority.Recipe.SourceModel, authority.Recipe.TargetModel,
		authority.Recipe.Task, compositionRuntimeResourcesFixture(t, authority),
	)
	if err != nil {
		t.Fatal(err)
	}
	publishCompositeGenerationPromotion(t, store, runtime.plan)
	direct, err := runtime.PromotedGeneration(context.Background(), store)
	if err != nil {
		t.Fatal(err)
	}
	surface, err := ResolveCompositeGenerationSurface(
		context.Background(), store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	)
	if err != nil || surface != direct || surface.Recipe != authority.Recipe.ID || surface.Plan != runtime.plan.ID {
		t.Fatalf("surface=%+v direct=%+v err=%v", surface, direct, err)
	}
}

func TestCompositeGenerationUnpromotedRefusal(t *testing.T) {
	store, authority := productionCompositionFixture(t, true)
	if _, err := ResolveCompositeGenerationSurface(
		context.Background(), store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	); err == nil {
		t.Fatal("active but generation-unpromoted composition reached a serving surface")
	}
}

func compositionRuntimeResourcesFixture(t *testing.T, authority composition.CompositionAuthority) CompositionRuntimeResources {
	t.Helper()
	source := &bridgeSourceFixture{
		model: authority.Recipe.SourceModel,
		value: inferenceBridgeValue(t, tensor.MustShape(2, 2), []float32{1, 2, 3, 4}),
	}
	target := &bridgeTargetFixture{
		model:     authority.Recipe.TargetModel,
		embedding: inferenceBridgeValue(t, tensor.MustShape(3, 2), []float32{5, 6, 7, 8, 9, 10}),
	}
	first := inferenceBridgeValue(t, tensor.MustShape(2, 3), []float32{1, 0, 0, 1, 1, 1})
	bias := inferenceBridgeValue(t, tensor.MustShape(3), []float32{1, 2, 3})
	return &bridgeRuntimeResourcesFixture{
		source: source, target: target,
		weights: RepresentationBridgeWeights{First: &first, FirstBias: &bias},
	}
}

func publishCompositeGenerationPromotion(
	t *testing.T,
	store *overgodb.Store,
	plan composition.CompositionExecutionPlan,
) {
	t.Helper()
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	output := id(artifact.KindOutput, "surface output")
	evidence, err := (composition.CompositeGenerationCUDAAuthority{}).New(composition.CompositeGenerationCUDAEvidence{
		SourceModel: plan.SourceModel, TargetModel: plan.TargetModel,
		CompositionRecipe: plan.CompositionRecipe, TargetBaselineRecipe: id(artifact.KindRecipe, "surface baseline"),
		ExecutionPlan: plan.ID, Promotion: plan.Promotion, Bridge: plan.BridgeWeights[0],
		HeldOutInputs:  []artifact.ID{id(artifact.KindOutput, "surface input")},
		BaselineOutput: output, ComposedOutput: output,
		BaselineRun: id(artifact.KindRun, "surface baseline run"), ComposedRun: id(artifact.KindRun, "surface composed run"),
		BaselineObservation: id(artifact.KindEvidence, "surface baseline observation"),
		ComposedObservation: id(artifact.KindEvidence, "surface composed observation"),
		Device:              id(artifact.KindEvidence, "surface device"), BridgeExecution: id(artifact.KindEvidence, "surface bridge execution"),
		ExactOutputParity: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	promotion, err := (composition.CompositeGenerationPromotionAuthority{}).Promote(plan, evidence)
	if err != nil {
		t.Fatal(err)
	}
	planContent, err := plan.Content()
	if err != nil {
		t.Fatal(err)
	}
	evidenceContent, err := evidence.Content()
	if err != nil {
		t.Fatal(err)
	}
	promotionContent, err := promotion.Content()
	if err != nil {
		t.Fatal(err)
	}
	alias, err := composition.CompositeGenerationPromotionAlias(plan.SourceModel, plan.TargetModel, plan.Task)
	if err != nil {
		t.Fatal(err)
	}
	lineage := append(append(plan.Lineage(), evidence.Lineage()...), promotion.Lineage()...)
	batch, err := artifact.NewDocumentBatch(
		"fixture/composite-generation/surface", []artifact.Content{planContent, evidenceContent, promotionContent},
		lineage, []artifact.AliasBinding{{Name: alias, Target: promotion.ID}},
	)
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[artifact.ID]bool)
	for _, edge := range lineage {
		if seen[edge.Parent] {
			continue
		}
		seen[edge.Parent] = true
		_, found, findErr := store.Artifact(context.Background(), edge.Parent)
		if findErr != nil {
			t.Fatal(findErr)
		}
		if !found && edge.Parent != plan.ID && edge.Parent != evidence.ID && edge.Parent != promotion.ID {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: edge.Parent, Size: 1})
		}
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		t.Fatal(err)
	}
}
