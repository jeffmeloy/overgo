package composition

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestCompositionResidencyPlan(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	ctx := context.Background()
	compile := func() (CompositionExecutionPlan, error) {
		return CompileCompositionExecutionPlan(
			ctx, store,
			authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
		)
	}
	if _, err := compile(); err == nil {
		t.Fatal("unactivated composition compiled")
	}
	batch, err := authority.Recipe.ActivationBatch(ctx, store, "fixture/composition/plan-activate", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	plan, err := compile()
	if err != nil {
		t.Fatal(err)
	}
	repeated, err := compile()
	if err != nil || repeated.ID != plan.ID || repeated.CacheIdentity != plan.CacheIdentity {
		t.Fatalf("repeated plan = %s/%s, want %s/%s: %v", repeated.ID, repeated.CacheIdentity, plan.ID, plan.CacheIdentity, err)
	}
	if plan.CompositionRecipe != authority.Recipe.ID ||
		plan.ExecutionRecipe != authority.Execution.ID ||
		plan.SourceModel != authority.Recipe.SourceModel || plan.TargetModel != authority.Recipe.TargetModel ||
		plan.SourceContract != authority.SourceContract.ID || plan.TargetContract != authority.TargetContract.ID ||
		plan.TrainingPolicy != authority.Recipe.TrainingPolicy ||
		plan.PromotionPolicy != authority.Recipe.PromotionPolicy || plan.Promotion != authority.Promotion.ID {
		t.Fatalf("plan authority = %+v", plan)
	}
	if len(plan.BridgeDefinitions) != tensor.SingletonExtent ||
		plan.BridgeDefinitions[tensor.FirstOffset] != authority.Bridge.ID ||
		len(plan.BridgeWeights) != tensor.SingletonExtent ||
		plan.BridgeWeights[tensor.FirstOffset] != authority.Bridge.Weights ||
		len(plan.Operators) != tensor.SingletonExtent ||
		plan.Operators[tensor.FirstOffset] != bridgegraph.OperatorLinear {
		t.Fatalf("ordered bridge program = %v, %v, %v", plan.BridgeDefinitions, plan.BridgeWeights, plan.Operators)
	}
	if plan.Capture.Model != authority.Recipe.SourceModel || plan.Capture.Contract != authority.SourceContract.ID ||
		plan.Injection.Model != authority.Recipe.TargetModel || plan.Injection.Contract != authority.TargetContract.ID {
		t.Fatalf("boundaries = %+v -> %+v", plan.Capture, plan.Injection)
	}
	if len(plan.Components) != tensor.PairedExtent ||
		plan.Components[tensor.FirstOffset].Module != modelrecipe.ModuleCaptureRepresentation ||
		plan.Components[tensor.SingletonExtent].Module != modelrecipe.ModuleInjectRepresentation ||
		plan.Components[tensor.FirstOffset].Placement != authority.Execution.Nodes[tensor.FirstOffset].Placement ||
		plan.Components[tensor.SingletonExtent].Placement != authority.Execution.Nodes[tensor.SingletonExtent].Placement ||
		plan.Components[tensor.FirstOffset].Residency != authority.Execution.Nodes[tensor.FirstOffset].Residency ||
		plan.Components[tensor.SingletonExtent].Residency != authority.Execution.Nodes[tensor.SingletonExtent].Residency ||
		plan.Components[tensor.FirstOffset].Lifetime != recipe.SessionCapacity ||
		plan.Components[tensor.SingletonExtent].Lifetime != recipe.SessionRequest {
		t.Fatalf("components = %+v", plan.Components)
	}
	content, err := plan.Content()
	if err != nil || content.Descriptor.ID != plan.ID {
		t.Fatalf("plan content = %+v, %v", content.Descriptor, err)
	}
	if _, err := CompileCompositionExecutionPlan(
		ctx, store, authority.Recipe.SourceModel,
		artifact.ID{}, authority.Recipe.Task,
	); err == nil {
		t.Fatal("invalid target scope compiled")
	}
	if _, err := CompileCompositionExecutionPlan(
		ctx, store, authority.Recipe.SourceModel,
		authority.Recipe.TargetModel, recipe.TaskEmbedding,
	); err == nil {
		t.Fatal("missing task-scoped composition compiled")
	}
}

func TestTransformedRepresentationCacheIdentity(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	ctx := context.Background()
	batch, err := authority.Recipe.ActivationBatch(ctx, store, "fixture/composition/cache-activate", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	plan, err := CompileCompositionExecutionPlan(
		ctx, store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	)
	if err != nil {
		t.Fatal(err)
	}
	firstInput := testutil.ArtifactID(t, artifact.KindOutput, "captured representation one")
	secondInput := testutil.ArtifactID(t, artifact.KindOutput, "captured representation two")
	first, err := plan.TransformedRepresentationCacheIdentity(firstInput)
	if err != nil {
		t.Fatal(err)
	}
	repeated, err := plan.TransformedRepresentationCacheIdentity(firstInput)
	if err != nil || repeated != first {
		t.Fatalf("repeated cache identity = %s, want %s: %v", repeated, first, err)
	}
	second, err := plan.TransformedRepresentationCacheIdentity(secondInput)
	if err != nil {
		t.Fatal(err)
	}
	if first == second {
		t.Fatal("distinct captured representations shared one transformed cache identity")
	}
	if _, err := plan.TransformedRepresentationCacheIdentity(artifact.ID{}); err == nil {
		t.Fatal("invalid captured representation admitted to transformed cache")
	}
	changed := plan
	changed.Components = append([]CompositionComponentPlan(nil), plan.Components...)
	changed.Components[tensor.FirstOffset].Lifetime = recipe.SessionRequest
	changed.ID = artifact.ID{}
	changed.CacheIdentity, err = compositionCacheIdentity(changed)
	if err != nil || changed.CacheIdentity == plan.CacheIdentity {
		t.Fatalf("lifetime-specific cache authority = %s, original %s: %v", changed.CacheIdentity, plan.CacheIdentity, err)
	}
}
