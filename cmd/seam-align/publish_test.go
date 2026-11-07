package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/jsonfile"
	"overgo/internal/mediacapability"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/rae"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAJudgedBridgePublishesAnInactiveComposite publishes the composite of a small encoder directory, a small text
// model and a bridge between their widths. Trials the policy refuses commit nothing. The recipe the store then holds
// loads with every authority it names; its models are the ones cmd/evaluate's composite resolves, the encoder as an
// image encoder and the text model as its runner, each definition one the store binds to its model; its promotion
// names the adapter, whose bytes the store keeps; no alias makes it active, and publishing it again commits nothing.
// Activated here alone, it compiles to a plan with both components on the card.
func TestAJudgedBridgePublishesAnInactiveComposite(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	work := t.TempDir()
	// The encoder gives a picture tokens of encoderWidth channels; the hermetic text model embeds in textWidth.
	const encoderWidth, textWidth, tokens = 4, 8, 5
	encoder := filepath.Join(work, "encoder")
	testcheck.NoError(t, os.MkdirAll(encoder, 0o700))
	testcheck.NoError(t, os.WriteFile(filepath.Join(encoder, "config.json"), []byte(`{"model_type":"dinov2","hidden_size":4}`), 0o600))
	testcheck.NoError(t, safetensors.Save(filepath.Join(encoder, "model.safetensors"),
		map[string][]float32{"embeddings.cls_token": {1, 2, 3, 4}}, map[string][]int{"embeddings.cls_token": {1, 1, encoderWidth}}, nil))
	text := testutil.HermeticLlamaGGUF(t, 64)
	bridge := rae.EncoderBridge{Weight: make([]float32, textWidth*encoderWidth), Bias: make([]float32, textWidth)}
	for index := range bridge.Weight {
		bridge.Weight[index] = float32(index%7) - 3
	}
	adapter := filepath.Join(work, "bridge.safetensors")
	testcheck.NoError(t, rae.SaveBridgeWeights(adapter, bridge))
	cited := map[string]string{
		"held-out.json": `{"cases":["held-out"]}`, "regression.json": `{"cases":["regression"]}`,
		"judgement.json": `{"verdict":"gain"}`, "run.json": `{"segments":2}`,
	}
	for name, body := range cited {
		testcheck.NoError(t, os.WriteFile(filepath.Join(work, name), []byte(body), 0o600))
	}
	trial := composition.RepresentationBridgePromotionTrial{
		Seed: 11, BridgeScore: 0.9, CheapBaselineScore: 0.8, DroppedSourceScore: 0.7, ShuffledSourceScore: 0.72,
		RegressionBaselineScore: 0.85, RegressionCandidateScore: 0.84, BaselineLatencyNS: 100, ComposedLatencyNS: 110,
		BaselinePeakDeviceBytes: 1000, ComposedPeakDeviceBytes: 1020,
	}
	again := trial
	again.Seed, again.BridgeScore = 29, 0.88
	spec := publishSpec{
		Encoder: encoder, TextModel: text, Adapter: adapter, EncoderTokens: tokens,
		HeldOutSplit: filepath.Join(work, "held-out.json"), RegressionSet: filepath.Join(work, "regression.json"),
		Evaluator: filepath.Join(work, "judgement.json"), TrainingPolicy: filepath.Join(work, "run.json"),
		Policy: composition.RepresentationBridgePromotionPolicy{
			Direction: runrecord.DirectionMaximize, MinimumSeeds: 2,
			MinimumHeldOutGain: 0.05, MinimumSourceDependence: 0.1, MaximumRegression: 0.02,
			MaximumSeedSpread: 0.05, MaximumLatencyIncrease: 0.25, MaximumDeviceByteIncrease: 64,
		},
		Trials: []composition.RepresentationBridgePromotionTrial{trial, again},
	}
	refused := spec
	short := again
	short.BridgeScore = trial.CheapBaselineScore
	refused.Trials = []composition.RepresentationBridgePromotionTrial{trial, short}
	specPath, refusedPath := filepath.Join(work, "publish.json"), filepath.Join(work, "refused.json")
	testcheck.NoError(t, jsonfile.Write(specPath, spec, 0o600))
	testcheck.NoError(t, jsonfile.Write(refusedPath, refused, 0o600))
	repository := filepath.Join(work, "store")
	head := func() artifact.CommitID {
		t.Helper()
		store := testcheck.Must(overgodb.Create(repository))(t)
		defer func() { testcheck.NoError(t, store.Close()) }()
		id, _ := store.Head()
		return id
	}
	started := head()
	if _, err := publishComposite(repository, refusedPath); err == nil {
		t.Fatal("trials whose bridge gains nothing over the cheap baseline were published")
	}
	if now := head(); now != started {
		t.Fatalf("a refused publication moved the store from %s to %s", started, now)
	}
	authority := testcheck.Must(publishComposite(repository, specPath))(t)
	published := head()
	if republished := testcheck.Must(publishComposite(repository, specPath))(t); republished.Recipe != authority.Recipe || head() != published {
		t.Fatalf("publishing the composite again gave %s and moved the store", republished.Recipe.ID)
	}
	store := overgodbtest.Open(t, repository)
	loaded := testcheck.Must(composition.LoadCompositionRecipe(ctx, store, authority.Recipe.ID))(t)
	if loaded != authority.Recipe {
		t.Fatalf("the store holds recipe %+v, the publication %+v", loaded, authority.Recipe)
	}
	source := testcheck.Must(mediacapability.Catalog[recipe.TaskImageEmbedding].Resolve(ctx, encoder))(t)
	target, description, err := modelcli.ResolveModel(ctx, store, text)
	testcheck.NoError(t, err)
	if loaded.SourceModel != source.Inventory.Manifest.ID || loaded.TargetModel != target {
		t.Fatalf("the recipe joins %s to %s, the serving path resolves %s and %s", loaded.SourceModel, loaded.TargetModel, source.Inventory.Manifest.ID, target)
	}
	encoderRecipe := testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskImageEmbedding, loaded.SourceModel))(t)
	testcheck.NoError(t, modelrecipe.RequireModelDefinitionBinding(ctx, store, authority.SourceContract.Producer.Definition, loaded.SourceModel, encoderRecipe.ID))
	if authority.TargetContract.Producer.Definition != description.Identity.Definition {
		t.Fatalf("the text model's contract names definition %s, its runner %s", authority.TargetContract.Producer.Definition, description.Identity.Definition)
	}
	testcheck.NoError(t, modelrecipe.RequireModelDefinitionBinding(ctx, store, description.Identity.Definition, target, description.Identity.Recipe))
	promotion := testcheck.Must(composition.LoadRepresentationBridgePromotion(ctx, store, loaded.Promotion))(t)
	bridgeID := testcheck.Must(artifact.IdentifyFile(artifact.KindAdapter, adapter))(t)
	content, found, readErr := artifact.ReadContent(ctx, store, bridgeID)
	if readErr != nil || !found || promotion.Bridge != bridgeID || loaded.BridgeWeights != bridgeID ||
		!bytes.Equal(content.Data, testcheck.Must(os.ReadFile(adapter))(t)) {
		t.Fatalf("the promotion names bridge %s, the adapter is %s, its bytes held %t (%v)", promotion.Bridge, bridgeID, found, readErr)
	}
	if _, active, err := composition.ActiveComposition(ctx, store, loaded.SourceModel, loaded.TargetModel, recipe.TaskGeneration); err != nil || active {
		t.Fatalf("the published candidate is active (%t, %v)", active, err)
	}
	testcheck.Must(store.Commit(ctx, testcheck.Must(loaded.ActivationBatch(ctx, store, "test/publish/activate", nil))(t)))(t)
	plan := testcheck.Must(composition.CompileCompositionExecutionPlan(ctx, store, loaded.SourceModel, loaded.TargetModel, recipe.TaskGeneration))(t)
	if plan.CompositionRecipe != loaded.ID {
		t.Fatalf("the activated plan compiles recipe %s, the publication %s", plan.CompositionRecipe, loaded.ID)
	}
	for _, component := range plan.Sessions.Components {
		if component.Placement != recipe.PlacementDevice {
			t.Fatalf("component %s runs at %s, not on the card", component.Model, component.Placement)
		}
	}
}
