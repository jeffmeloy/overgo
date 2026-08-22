package inference

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/composition"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/representation"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type bridgeSourceFixture struct {
	model  artifact.ID
	value  reference.Value
	layers []int32
	calls  int
}

func (source *bridgeSourceFixture) ModelID() artifact.ID { return source.model }

func (source *bridgeSourceFixture) ExtractLayerInputs(
	_ context.Context,
	_ []tokenizer.TokenID,
	layers []int32,
) (reference.Value, error) {
	source.calls++
	source.layers = slices.Clone(layers)
	return source.value.Clone(), nil
}

type bridgeTargetFixture struct {
	model     artifact.ID
	embedding reference.Value
	overrides []EmbeddingOverride
}

type bridgeRuntimeResourcesFixture struct {
	source     RepresentationSource
	target     RepresentationTarget
	weights    RepresentationBridgeWeights
	definition artifact.ID
	authority  composition.BridgeWeightAuthority
	calls      int
}

func (resources *bridgeRuntimeResourcesFixture) Source(
	_ context.Context,
	_ composition.CompositionComponentPlan,
) (RepresentationSource, error) {
	resources.calls++
	return resources.source, nil
}

func (resources *bridgeRuntimeResourcesFixture) Target(
	_ context.Context,
	_ composition.CompositionComponentPlan,
) (RepresentationTarget, error) {
	resources.calls++
	return resources.target, nil
}

func (resources *bridgeRuntimeResourcesFixture) BridgeWeights(
	_ context.Context,
	definition artifact.ID,
	authority composition.BridgeWeightAuthority,
) (RepresentationBridgeWeights, error) {
	resources.calls++
	resources.definition, resources.authority = definition, authority
	return resources.weights, nil
}

func (target *bridgeTargetFixture) ModelID() artifact.ID { return target.model }

func (target *bridgeTargetFixture) ForwardWithEmbeddingOverrides(
	_ context.Context,
	_ []tokenizer.TokenID,
	overrides []EmbeddingOverride,
) (reference.Value, error) {
	target.overrides = slices.Clone(overrides)
	result := target.embedding.Clone()
	return result, applyEmbeddingOverrides(&result, overrides)
}

func TestProductionEmbeddingInjection(t *testing.T) {
	store, authority := productionCompositionFixture(t, true)
	sourceModel, targetModel := authority.Recipe.SourceModel, authority.Recipe.TargetModel
	source := &bridgeSourceFixture{
		model: sourceModel,
		value: inferenceBridgeValue(t, tensor.MustShape(2, 2), []float32{1, 2, 3, 4}),
	}
	target := &bridgeTargetFixture{
		model:     targetModel,
		embedding: inferenceBridgeValue(t, tensor.MustShape(3, 2), []float32{0, 0, 0, 0, 0, 0}),
	}
	first := reference.ZeroValue(tensor.MustShape(2, 3))
	bias := inferenceBridgeValue(t, tensor.MustShape(3), []float32{10, 20, 30})
	resources := &bridgeRuntimeResourcesFixture{
		source: source, target: target,
		weights: RepresentationBridgeWeights{First: &first, FirstBias: &bias},
	}
	runtime, err := OpenProductionComposition(
		context.Background(), store, sourceModel, targetModel, authority.Recipe.Task, resources,
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := runtime.Forward(
		context.Background(),
		[]tokenizer.TokenID{1, 2}, []tokenizer.TokenID{3, 4},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{10, 20, 30, 10, 20, 30}
	if !slices.Equal(result.Data, want) || len(target.overrides) != len(want)/len(bias.Data) ||
		!slices.Equal(source.layers, []int32{int32(*authority.SourceContract.Producer.Layer)}) ||
		resources.definition != authority.Bridge.ID || resources.authority.Weights.ID != authority.Bridge.Weights ||
		resources.authority.Inventory.ID != authority.Bridge.WeightInventory || runtime.PlanIdentity().Kind() != artifact.KindProfile {
		t.Fatalf("bridge result=%v overrides=%v layers=%v", result.Data, target.overrides, source.layers)
	}
	wrongTarget := *target
	wrongTarget.model = sourceModel
	resources.target = &wrongTarget
	if _, err := OpenProductionComposition(
		context.Background(), store, sourceModel, targetModel, authority.Recipe.Task, resources,
	); err == nil {
		t.Fatal("mismatched target model accepted")
	}
}

func TestNoDirectCompositionConstructors(t *testing.T) {
	store, authority := productionCompositionFixture(t, false)
	resources := &bridgeRuntimeResourcesFixture{}
	if _, err := OpenProductionComposition(
		context.Background(), store,
		authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
		resources,
	); err == nil || resources.calls != 0 {
		t.Fatalf("inactive composition reached runtime resources: calls=%d err=%v", resources.calls, err)
	}
	if _, err := (&ProductionComposition{}).Forward(context.Background(), nil, nil); err == nil {
		t.Fatal("zero-value production composition executed")
	}
}

func TestTransformedRepresentationCacheIdentity(t *testing.T) {
	store, authority := productionCompositionFixture(t, true)
	source := &bridgeSourceFixture{
		model: authority.Recipe.SourceModel,
		value: inferenceBridgeValue(t, tensor.MustShape(2, 2), []float32{1, 2, 3, 4}),
	}
	target := &bridgeTargetFixture{
		model:     authority.Recipe.TargetModel,
		embedding: inferenceBridgeValue(t, tensor.MustShape(3, 2), []float32{0, 0, 0, 0, 0, 0}),
	}
	first := reference.ZeroValue(tensor.MustShape(2, 3))
	bias := inferenceBridgeValue(t, tensor.MustShape(3), []float32{10, 20, 30})
	resources := &bridgeRuntimeResourcesFixture{
		source: source, target: target,
		weights: RepresentationBridgeWeights{First: &first, FirstBias: &bias},
	}
	runtime, err := OpenProductionComposition(
		context.Background(), store,
		authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task, resources,
	)
	if err != nil {
		t.Fatal(err)
	}
	sourceTokens := []tokenizer.TokenID{1, 2}
	targetTokens := []tokenizer.TokenID{3, 4}
	identity, err := runtime.TransformedRepresentationCacheIdentity(sourceTokens, targetTokens)
	if err != nil {
		t.Fatal(err)
	}
	repeated, err := runtime.TransformedRepresentationCacheIdentity(sourceTokens, targetTokens)
	if err != nil || repeated != identity {
		t.Fatalf("repeated cache identity = %s, want %s: %v", repeated, identity, err)
	}
	changed, err := runtime.TransformedRepresentationCacheIdentity(
		[]tokenizer.TokenID{2, 1}, targetTokens,
	)
	if err != nil || changed == identity {
		t.Fatalf("changed cache identity = %s, original %s: %v", changed, identity, err)
	}
	if _, err := runtime.Forward(context.Background(), sourceTokens, targetTokens); err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.Forward(context.Background(), sourceTokens, targetTokens); err != nil {
		t.Fatal(err)
	}
	if source.calls != tensor.SingletonExtent {
		t.Fatalf("immutable transformed representation recomputed %d times", source.calls)
	}
}

func productionCompositionFixture(t *testing.T, activate bool) (*repodb.Store, composition.CompositionAuthority) {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	ctx := context.Background()
	sourceModel := testutil.ArtifactID(t, artifact.KindModel, "bridge-source-model")
	targetModel := testutil.ArtifactID(t, artifact.KindModel, "bridge-target-model")
	layer := uint32(0)
	_, sourceContract := inferenceBridgeContract(
		t, sourceModel, representation.TapLayerInput, &layer, representation.ModalityAudio, 2,
	)
	_, targetContract := inferenceBridgeContract(
		t, targetModel, representation.TapEmbeddingOutput, nil, representation.ModalityText, 3,
	)
	weights := testutil.ArtifactID(t, artifact.KindAdapter, "bridge-weights")
	inventory := testutil.ArtifactID(t, artifact.KindTensorInventory, "bridge-inventory")
	heldOut := testutil.ArtifactID(t, artifact.KindDatasetShard, "bridge-held-out")
	regression := testutil.ArtifactID(t, artifact.KindDatasetShard, "bridge-regression")
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "bridge-evaluator")
	trainingPolicy := testutil.ArtifactID(t, artifact.KindProfile, "bridge-training-policy")
	promotionPolicy, err := composition.NewRepresentationBridgePromotionPolicy(composition.RepresentationBridgePromotionPolicy{
		Direction: runrecord.DirectionMaximize, MinimumSeeds: 2,
		MinimumHeldOutGain: 0.05, MinimumSourceDependence: 0.1, MaximumRegression: 0.02,
		MaximumSeedSpread: 0.05, MaximumLatencyIncrease: 0.25, MaximumDeviceByteIncrease: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	parents := []artifact.ID{
		sourceModel, targetModel, sourceContract.Producer.Definition, targetContract.Producer.Definition,
		weights, inventory, heldOut, regression, evaluator, trainingPolicy,
	}
	descriptors := make([]artifact.Descriptor, len(parents))
	for index, id := range parents {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/production-composition/parents", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	bridge, err := composition.NewBridgeDefinition(composition.BridgeDefinition{
		SourceModel: sourceModel, TargetModel: targetModel,
		Graph: bridgegraph.Definition{
			Source: sourceContract.ID, Target: targetContract.ID,
			Operator: bridgegraph.OperatorLinear, Bias: true,
		},
		Weights: weights, WeightInventory: inventory,
	}, sourceContract, targetContract)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := (modelrecipe.RepresentationBridgeCompiler{
		SourcePlacement: recipe.PlacementDevice, TargetPlacement: recipe.PlacementDevice,
		SourceResidency: recipe.ResidencyDeviceF32, TargetResidency: recipe.ResidencyDeviceF32,
		SourceSession: recipe.SessionCapacity, TargetSession: recipe.SessionRequest,
	}).Definition(sourceModel, targetModel, sourceContract.ID, targetContract.ID, weights)
	if err != nil {
		t.Fatal(err)
	}
	promotion, err := (composition.RepresentationBridgePromoter{}).Evaluate(promotionPolicy, composition.RepresentationBridgePromotion{
		Bridge: weights, SourceModel: sourceModel, TargetModel: targetModel,
		SourceContract: sourceContract.ID, TargetContract: targetContract.ID,
		HeldOutSplit: heldOut, RegressionSet: regression, Evaluator: evaluator,
		Trials: []composition.RepresentationBridgePromotionTrial{
			{Seed: 11, BridgeScore: 0.9, CheapBaselineScore: 0.8, DroppedSourceScore: 0.7, ShuffledSourceScore: 0.72, RegressionBaselineScore: 0.85, RegressionCandidateScore: 0.84, BaselineLatencyNS: 100, ComposedLatencyNS: 110, BaselinePeakDeviceBytes: 1000, ComposedPeakDeviceBytes: 1020},
			{Seed: 29, BridgeScore: 0.88, CheapBaselineScore: 0.8, DroppedSourceScore: 0.69, ShuffledSourceScore: 0.7, RegressionBaselineScore: 0.86, RegressionCandidateScore: 0.85, BaselineLatencyNS: 100, ComposedLatencyNS: 112, BaselinePeakDeviceBytes: 1000, ComposedPeakDeviceBytes: 1024},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	compositionRecipe, err := composition.NewCompositionRecipe(composition.CompositionRecipe{
		SourceModel: sourceModel, TargetModel: targetModel, Task: recipe.TaskGeneration,
		SourceContract: sourceContract.ID, TargetContract: targetContract.ID,
		BridgeDefinition: bridge.ID, BridgeWeights: weights, ExecutionRecipe: execution.ID,
		TrainingPolicy: trainingPolicy, PromotionPolicy: promotionPolicy.ID, Promotion: promotion.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	authority := composition.CompositionAuthority{
		SourceContract: sourceContract, TargetContract: targetContract, Bridge: bridge,
		Execution: execution, PromotionPolicy: promotionPolicy, Promotion: promotion, Recipe: compositionRecipe,
	}
	batch, err := authority.Batch("fixture/production-composition/authority")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if activate {
		activation, err := authority.Recipe.ActivationBatch(ctx, store, "fixture/production-composition/activate", nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, activation); err != nil {
			t.Fatal(err)
		}
	}
	return store, authority
}

func inferenceBridgeContract(
	t *testing.T,
	modelID artifact.ID,
	tap representation.TapPoint,
	layer *uint32,
	modality representation.Modality,
	width uint64,
) ([]byte, representation.Contract) {
	t.Helper()
	value := representation.Contract{
		Version: representation.ContractVersion,
		Producer: representation.Producer{
			Model:      modelID,
			Definition: testutil.ArtifactID(t, artifact.KindModelDefinition, string(modality)+"-definition"),
			Tap:        tap, Layer: layer,
		},
		Modality: modality,
		Tensor: representation.TensorContract{DataType: dtype.F32, Axes: []representation.Axis{
			{Kind: representation.AxisChannel, Bounds: representation.AxisBounds{Extent: width}},
			{Kind: representation.AxisSequence, Bounds: representation.AxisBounds{Minimum: 1, Maximum: 4}},
		}},
		Sequence: representation.SequenceContract{
			Axis: representation.AxisSequence, Mask: representation.MaskPrefix,
			Padding: representation.PaddingSuffix, Position: representation.PositionSequential,
			PositionAxes: []representation.AxisKind{representation.AxisSequence},
		},
		Normalization: representation.NormalizationContract{
			Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative,
		},
	}
	content, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := representation.ParseContract(content)
	if err != nil {
		t.Fatal(err)
	}
	return content, parsed
}

func inferenceBridgeValue(t *testing.T, shape tensor.Shape, data []float32) reference.Value {
	t.Helper()
	value, err := reference.NewValue(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	return value
}
