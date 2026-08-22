package composition

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/tensor"
)

func TestExternalCrossAttentionPlan(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	ctx := context.Background()
	sourceChannels := authority.SourceContract.Tensor.Axes[tensor.FirstOffset].Bounds.Extent
	targetChannels := authority.TargetContract.Tensor.Axes[tensor.FirstOffset].Bounds.Extent
	sourceTokenLimit := authority.SourceContract.Tensor.Axes[tensor.SingletonExtent].Bounds.Maximum
	layer := *authority.SourceContract.Producer.Layer
	headCount := uint64(tensor.PairedExtent)
	external, err := NewExternalCrossAttentionDefinition(ExternalCrossAttentionDefinition{
		Target: authority.Recipe.TargetModel, Source: authority.Recipe.SourceModel,
		Adapter: authority.Recipe.BridgeWeights, Layers: []uint32{layer},
		SourceChannels: sourceChannels, HeadCount: headCount, SourceTokenLimit: sourceTokenLimit,
	})
	if err != nil {
		t.Fatal(err)
	}
	authority.Recipe.ExternalCrossAttention = external.ID
	authority.Recipe, err = NewCompositionRecipe(authority.Recipe)
	if err != nil {
		t.Fatal(err)
	}
	authority.ExternalCrossAttention = &external
	batch, err := authority.Batch("fixture/composition/external-cross-attention")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	profile, found := model.LookupArchitecture("llama")
	if !found {
		t.Fatal("llama architecture profile is absent")
	}
	keyChannels := targetChannels / headCount
	plan, err := model.CompileModelPlanWithProfile(model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: profile.Name, ContextLength: uint32(sourceTokenLimit),
			EmbeddingLength: uint32(targetChannels), BlockCount: layer + uint32(tensor.SingletonExtent),
			VocabularySize: uint32(sourceChannels + targetChannels),
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: uint32(headCount), HeadCountKV: uint32(headCount),
			KeyLength: uint32(keyChannels), ValueLength: uint32(keyChannels),
		},
	}, model.Weights{}, profile)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileExternalCrossAttentionPlan(
		ctx, store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task, plan,
	); err == nil {
		t.Fatal("external cross-attention compiled before recipe activation")
	}
	activation, err := authority.Recipe.ActivationBatch(ctx, store, "fixture/composition/external-activate", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, activation); err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileExternalCrossAttentionPlan(
		ctx, store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task, plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	if compiled.CompositionRecipe != authority.Recipe.ID || compiled.Definition != external.ID ||
		compiled.Adapter != authority.Recipe.BridgeWeights || compiled.CacheIdentity.Kind() != artifact.KindProfile ||
		compiled.TargetChannels != targetChannels || compiled.HeadChannels != keyChannels ||
		len(compiled.Layers) != tensor.SingletonExtent || compiled.Layers[tensor.FirstOffset] != layer {
		t.Fatalf("external cross-attention plan = %+v", compiled)
	}
}
