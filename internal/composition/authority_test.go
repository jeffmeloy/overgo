package composition

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/representation"
	"overgo/internal/runrecord"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testutil"
)

func TestRepresentationContractRepository(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	loaded, err := representation.LoadContract(context.Background(), store, authority.SourceContract.ID)
	if err != nil || loaded.ID != authority.SourceContract.ID || loaded.Producer.Model != authority.Recipe.SourceModel {
		t.Fatalf("loaded source contract = %+v, %v", loaded, err)
	}
	if _, err := representation.LoadContract(context.Background(), store, authority.Bridge.ID); err == nil {
		t.Fatal("bridge definition admitted as a representation contract")
	}
}

func TestBridgeDefinitionRepository(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	loaded, err := LoadBridgeDefinition(context.Background(), store, authority.Bridge.ID)
	if err != nil || loaded != authority.Bridge || loaded.Weights != authority.Recipe.BridgeWeights {
		t.Fatalf("loaded bridge definition = %+v, %v", loaded, err)
	}
	weights, err := LoadBridgeWeights(context.Background(), store, authority.Bridge.ID)
	if err != nil || weights.Weights.ID != authority.Bridge.Weights ||
		weights.Inventory.ID != authority.Bridge.WeightInventory {
		t.Fatalf("loaded bridge weights = %+v, %v", weights, err)
	}
}

func TestCompositionRecipeRepository(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	loaded, err := LoadCompositionRecipe(context.Background(), store, authority.Recipe.ID)
	if err != nil || loaded != authority.Recipe || loaded.Promotion != authority.Promotion.ID {
		t.Fatalf("loaded composition recipe = %+v, %v", loaded, err)
	}
}

func TestActiveCompositionAlias(t *testing.T) {
	store, authority := compositionAuthorityFixture(t)
	ctx := context.Background()
	if _, found, err := ActiveComposition(
		ctx, store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	); err != nil || found {
		t.Fatalf("pre-activation lookup = %v, %v", found, err)
	}
	batch, err := authority.Recipe.ActivationBatch(ctx, store, "fixture/composition/activate", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	active, found, err := ActiveComposition(
		ctx, store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	)
	if err != nil || !found || active != authority.Recipe {
		t.Fatalf("active composition = %+v, %v, %v", active, found, err)
	}
	if _, err := authority.Recipe.ActivationBatch(ctx, store, "fixture/composition/stale-activate", nil); err == nil {
		t.Fatal("activation replaced an existing alias without compare-and-set authority")
	}
}

func compositionAuthorityFixture(t *testing.T) (*repodb.Store, CompositionAuthority) {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	ctx := context.Background()
	sourceModel := testutil.ArtifactID(t, artifact.KindModel, "composition source model")
	targetModel := testutil.ArtifactID(t, artifact.KindModel, "composition target model")
	sourceDefinition := testutil.ArtifactID(t, artifact.KindModelDefinition, "composition source definition")
	targetDefinition := testutil.ArtifactID(t, artifact.KindModelDefinition, "composition target definition")
	weights := testutil.ArtifactID(t, artifact.KindAdapter, "composition bridge weights")
	inventory := testutil.ArtifactID(t, artifact.KindTensorInventory, "composition bridge inventory")
	heldOut := testutil.ArtifactID(t, artifact.KindDatasetShard, "composition held-out split")
	regression := testutil.ArtifactID(t, artifact.KindDatasetShard, "composition regression split")
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "composition evaluator")
	parents := []artifact.ID{
		sourceModel, targetModel, sourceDefinition, targetDefinition,
		weights, inventory, heldOut, regression, evaluator,
	}
	descriptors := make([]artifact.Descriptor, len(parents))
	for index, id := range parents {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/composition/parents", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}

	sequence := representation.SequenceContract{
		Axis: representation.AxisSequence, Mask: representation.MaskPrefix,
		Padding: representation.PaddingSuffix, Position: representation.PositionSequential,
		PositionAxes: []representation.AxisKind{representation.AxisSequence},
	}
	contract := func(model, definition artifact.ID, tap representation.TapPoint, layer *uint32, channels uint64) representation.Contract {
		value, err := representation.NewContract(representation.Contract{
			Producer: representation.Producer{Model: model, Definition: definition, Tap: tap, Layer: layer},
			Modality: representation.ModalityText,
			Tensor: representation.TensorContract{DataType: dtype.F32, Axes: []representation.Axis{
				{Kind: representation.AxisChannel, Bounds: representation.AxisBounds{Extent: channels}},
				{Kind: representation.AxisSequence, Bounds: representation.AxisBounds{Minimum: 1, Maximum: 16}},
			}},
			Sequence: sequence,
			Normalization: representation.NormalizationContract{
				Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	layer := uint32(3)
	source := contract(sourceModel, sourceDefinition, representation.TapLayerInput, &layer, 4)
	target := contract(targetModel, targetDefinition, representation.TapEmbeddingOutput, nil, 6)
	graph := bridgegraph.Definition{Source: source.ID, Target: target.ID, Operator: bridgegraph.OperatorLinear}
	bridge, err := NewBridgeDefinition(BridgeDefinition{
		SourceModel: sourceModel, TargetModel: targetModel, Graph: graph,
		Weights: weights, WeightInventory: inventory,
	}, source, target)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := (modelrecipe.RepresentationBridgeCompiler{
		SourcePlacement: recipe.PlacementHost, TargetPlacement: recipe.PlacementHost,
		SourceResidency: recipe.ResidencyHostCache, TargetResidency: recipe.ResidencyHostCache,
		SourceSession: recipe.SessionCapacity, TargetSession: recipe.SessionRequest,
	}).Definition(sourceModel, targetModel, source.ID, target.ID, weights)
	if err != nil {
		t.Fatal(err)
	}
	promotion, err := (RepresentationBridgePromoter{}).Evaluate(RepresentationBridgePromotion{
		Bridge: weights, SourceModel: sourceModel, TargetModel: targetModel,
		SourceContract: source.ID, TargetContract: target.ID,
		HeldOutSplit: heldOut, RegressionSet: regression, Evaluator: evaluator,
		Policy: RepresentationBridgePromotionPolicy{
			Direction: runrecord.DirectionMaximize, MinimumSeeds: 2,
			MinimumHeldOutGain: 0.05, MinimumSourceDependence: 0.1, MaximumRegression: 0.02,
		},
		Trials: []RepresentationBridgePromotionTrial{
			{Seed: 11, BridgeScore: 0.9, CheapBaselineScore: 0.8, DroppedSourceScore: 0.7, ShuffledSourceScore: 0.72, RegressionBaselineScore: 0.85, RegressionCandidateScore: 0.84},
			{Seed: 29, BridgeScore: 0.88, CheapBaselineScore: 0.8, DroppedSourceScore: 0.69, ShuffledSourceScore: 0.7, RegressionBaselineScore: 0.86, RegressionCandidateScore: 0.85},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	compositionRecipe, err := NewCompositionRecipe(CompositionRecipe{
		SourceModel: sourceModel, TargetModel: targetModel, Task: recipe.TaskGeneration,
		SourceContract: source.ID, TargetContract: target.ID,
		BridgeDefinition: bridge.ID, BridgeWeights: weights,
		ExecutionRecipe: execution.ID, Promotion: promotion.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	authority := CompositionAuthority{
		SourceContract: source, TargetContract: target, Bridge: bridge,
		Execution: execution, Promotion: promotion, Recipe: compositionRecipe,
	}
	batch, err := authority.Batch("fixture/composition/authority")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	return store, authority
}
