package inference

import (
	"context"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
)

func TestExternalCacheSeparateFromTargetKV(t *testing.T) {
	const externalTokens = uint64(2)
	spec := model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: "llama", ContextLength: 4, EmbeddingLength: 2,
			BlockCount: 2, VocabularySize: 8,
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: 1, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		},
	}
	plan, err := fixtureModelPlan(spec, model.Weights{})
	if err != nil {
		t.Fatal(err)
	}
	targetID := testutil.ArtifactID(t, artifact.KindModel, "external-target")
	sourceID := testutil.ArtifactID(t, artifact.KindModel, "external-source")
	adapterID := testutil.ArtifactID(t, artifact.KindAdapter, "external-adapter")
	compiled := composition.ExternalCrossAttentionPlan{
		Execution:   testutil.ArtifactID(t, artifact.KindProfile, "external-execution"),
		SourceModel: sourceID, TargetModel: targetID, Adapter: adapterID,
		Layer: 1, SourceChannels: 2, TargetChannels: 2, HeadCount: 1, SourceTokenLimit: 3,
		ID: testutil.ArtifactID(t, artifact.KindProfile, "external-plan"),
	}
	program, err := (ExternalCrossAttentionCompiler{}).Compile(compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	identity := inferenceBridgeValue(t, tensor.MustShape(2, 2), []float32{1, 0, 0, 1})
	weights := ExternalCrossAttentionWeights{
		Query: identity, Key: identity, Value: identity, Output: identity,
	}
	target := inferenceBridgeValue(t, tensor.MustShape(2, 2), []float32{1, 2, 3, 4})
	source := inferenceBridgeValue(t, tensor.MustShape(2, 2), []float32{5, 6, 7, 8})
	targetCache := &KVCache{
		Layers: []LayerCache{{
			Key:   reference.ZeroValue(tensor.MustShape(2, 1)),
			Value: reference.ZeroValue(tensor.MustShape(2, 1)),
		}},
		Tokens: 1, Position: 1,
	}
	cacheBefore := cloneCache(targetCache)
	output, external, err := program.Apply(context.Background(), 1, ExternalCrossAttentionInput{
		Target: target, TargetTokens: 2, Source: &source, SourceTokens: 2,
	}, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(output.Data, target.Data) {
		t.Fatalf("zero-gated output=%v want identity=%v", output.Data, target.Data)
	}
	if external.Program != program.Identity() || external.Source != sourceID ||
		external.Tokens != externalTokens || !reflect.DeepEqual(targetCache, cacheBefore) {
		t.Fatalf("external/target cache isolation failed: external=%+v target=%+v", external, targetCache)
	}
	weights.Gate = 1
	conditioned, reused, err := program.Apply(context.Background(), 1, ExternalCrossAttentionInput{
		Target: target, TargetTokens: 2,
	}, weights, &external)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(conditioned.Data, target.Data) || !reflect.DeepEqual(reused, external) ||
		!reflect.DeepEqual(targetCache, cacheBefore) {
		t.Fatalf("cached external attention=%v cache=%+v target-cache=%+v", conditioned.Data, reused, targetCache)
	}
	if _, _, err := program.Apply(context.Background(), 0, ExternalCrossAttentionInput{
		Target: target, TargetTokens: 2,
	}, weights, &external); err == nil {
		t.Fatal("undeclared target layer seam accepted")
	}
}
