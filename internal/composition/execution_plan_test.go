package composition

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/tensor"
)

func TestCompositionExecutionPlan(t *testing.T) {
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
	if plan.Sessions.Recipe != authority.Execution.ID || len(plan.Sessions.Components) != tensor.PairedExtent ||
		plan.Sessions.Components[tensor.FirstOffset].Module != modelrecipe.ModuleCaptureRepresentation ||
		plan.Sessions.Components[tensor.SingletonExtent].Module != modelrecipe.ModuleInjectRepresentation ||
		plan.Sessions.Components[tensor.FirstOffset].Session != recipe.SessionCapacity ||
		plan.Sessions.Components[tensor.SingletonExtent].Session != recipe.SessionRequest {
		t.Fatalf("sessions = %+v", plan.Sessions)
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

func TestCompositionResidencyPlan(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	ctx := context.Background()
	batch, err := authority.Recipe.ActivationBatch(ctx, store, "fixture/composition/residency-activate", nil)
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
	for _, component := range plan.Sessions.Components {
		if component.Placement != recipe.PlacementDevice || component.Residency != recipe.ResidencyDeviceF32 {
			t.Fatalf("non-resident composition component = %+v", component)
		}
	}
	if plan.Sessions.Components[tensor.FirstOffset].Session != recipe.SessionCapacity ||
		plan.Sessions.Components[tensor.SingletonExtent].Session != recipe.SessionRequest {
		t.Fatalf("composition lifetimes = %+v", plan.Sessions)
	}
}
