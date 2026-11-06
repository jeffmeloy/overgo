package inference

import (
	"context"
	"encoding/json"
	"image"
	"image/color"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/composition"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// bridgeEncoderChannels and bridgeEncoderTokens are the fixture encoder's output: three channels for each of four
// tokens.
const (
	bridgeEncoderChannels = 3
	bridgeEncoderTokens   = 4
)

// bridgeSourceFixture encodes an image as its colour means, a token for each step of a ramp: an encoder whose output
// follows its picture.
type bridgeSourceFixture struct {
	model artifact.ID
	calls int
}

func (source *bridgeSourceFixture) ModelID() artifact.ID { return source.model }

func (source *bridgeSourceFixture) EncodeImage(_ context.Context, picture image.Image) (reference.Value, error) {
	source.calls++
	var means [bridgeEncoderChannels]float64
	bounds := picture.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			red, green, blue, _ := picture.At(x, y).RGBA()
			means[0], means[1], means[2] = means[0]+float64(red), means[1]+float64(green), means[2]+float64(blue)
		}
	}
	pixels := float64(bounds.Dx() * bounds.Dy() * math.MaxUint16)
	data := make([]float32, bridgeEncoderChannels*bridgeEncoderTokens)
	for token := range bridgeEncoderTokens {
		for channel := range bridgeEncoderChannels {
			data[token*bridgeEncoderChannels+channel] = float32(means[channel] / pixels * float64(token+1))
		}
	}
	return reference.NewValue(tensor.MustShape(bridgeEncoderChannels, bridgeEncoderTokens), data)
}

// bridgeTargetFixture stands for a text model by its identity, its width and its image placeholder.
type bridgeTargetFixture struct {
	model       artifact.ID
	width       int
	placeholder string
}

func (target bridgeTargetFixture) ModelID() artifact.ID { return target.model }

func (target bridgeTargetFixture) EmbeddingWidth() int { return target.width }

func (target bridgeTargetFixture) ImagePlaceholder() string { return target.placeholder }

type bridgeRuntimeResourcesFixture struct {
	source  RepresentationSource
	target  RepresentationTarget
	weights RepresentationBridgeWeights
	calls   int
}

func (resources *bridgeRuntimeResourcesFixture) Source(context.Context, modelrecipe.ComponentSession) (RepresentationSource, error) {
	resources.calls++
	return resources.source, nil
}

func (resources *bridgeRuntimeResourcesFixture) Target(context.Context, modelrecipe.ComponentSession) (RepresentationTarget, error) {
	resources.calls++
	return resources.target, nil
}

func (resources *bridgeRuntimeResourcesFixture) BridgeWeights(
	context.Context, modelrecipe.ComponentSession, artifact.ID, composition.BridgeWeightAuthority,
) (RepresentationBridgeWeights, error) {
	resources.calls++
	return resources.weights, nil
}

// TestACompositionServesThroughARecipe holds an image composite to serving through its active composition recipe on
// the host: an inactive recipe reaches no resource and a bridge whose width differs from the text model's opens
// nothing, while the active one turns a picture into the bridge's rows, the bias plus the encoder's tokens through its
// map, at as many placeholder tokens of the prompt, which a host runner answers through the placeholder path; another
// picture gives other rows.
func TestACompositionServesThroughARecipe(t *testing.T) {
	t.Parallel()
	// The fixture's vocabulary holds no image token, so one of its control tokens stands for one.
	const placeholder = "</s>"
	runner := testcheck.Must(openFixtureRunnerWithResidency(testutil.HermeticLlamaGGUF(t, 64), OpenOptions{}, recipe.ResidencyHostReference))(t)
	t.Cleanup(func() { _ = runner.Close() })
	width := int(runner.spec.EmbeddingLength)
	first := make([]float32, bridgeEncoderChannels*width)
	bias := make([]float32, width)
	for index := range first {
		first[index] = float32(index%7) - 3
	}
	for index := range bias {
		bias[index] = float32(index) / 8
	}
	firstValue := testcheck.Must(reference.NewValue(tensor.MustShape(bridgeEncoderChannels, uint64(width)), first))(t)
	biasValue := testcheck.Must(reference.NewValue(tensor.MustShape(uint64(width)), bias))(t)
	weights := RepresentationBridgeWeights{First: &firstValue, FirstBias: &biasValue}
	open := func(activate bool, targetWidth int) (*ProductionComposition, *bridgeRuntimeResourcesFixture, error) {
		store, authority := productionCompositionFixture(t, activate, runner.ModelID(), uint64(width))
		resources := &bridgeRuntimeResourcesFixture{
			source:  &bridgeSourceFixture{model: authority.Recipe.SourceModel},
			target:  bridgeTargetFixture{model: runner.ModelID(), width: targetWidth, placeholder: placeholder},
			weights: weights,
		}
		composite, err := OpenProductionComposition(t.Context(), store, authority.Recipe.SourceModel, runner.ModelID(), authority.Recipe.Task, resources)
		return composite, resources, err
	}
	if _, resources, err := open(false, width); err == nil || resources.calls != 0 {
		t.Fatalf("an inactive composition reached its resources %d times (%v)", resources.calls, err)
	}
	if _, _, err := open(true, width-1); err == nil {
		t.Fatal("a bridge wider than the text model's embeddings opened")
	}
	composite, _, err := open(true, width)
	testcheck.NoError(t, err)
	if !composite.Recipe().Valid() || !composite.Capabilities().Image {
		t.Fatalf("the composite names no recipe or takes no image: %+v", composite.Capabilities())
	}
	rows := func(fill color.RGBA) ([]float32, []float32) {
		picture := image.NewRGBA(image.Rect(0, 0, 2, 2))
		for index := range picture.Pix {
			picture.Pix[index] = [4]uint8{fill.R, fill.G, fill.B, fill.A}[index%4]
		}
		prompt := testcheck.Must(composite.BuildImagePrompt(t.Context(), runner, picture, "", "ab", false))(t)
		tokens, inputs, err := CompileProjectedInputs(prompt, uint32(width))
		testcheck.NoError(t, err)
		if len(prompt.EmbeddingTokenIndices) != bridgeEncoderTokens || len(inputs.EmbeddingOverrides) != bridgeEncoderTokens {
			t.Fatalf("the picture took %d placeholders, want one for each of the encoder's %d tokens", len(prompt.EmbeddingTokenIndices), bridgeEncoderTokens)
		}
		encoded := testcheck.Must((&bridgeSourceFixture{}).EncodeImage(t.Context(), picture))(t)
		want := make([]float32, 0, len(prompt.Embeddings))
		for token := range bridgeEncoderTokens {
			for column := range width {
				value := bias[column]
				for channel := range bridgeEncoderChannels {
					value += encoded.Data[token*bridgeEncoderChannels+channel] * first[column*bridgeEncoderChannels+channel]
				}
				want = append(want, value)
			}
		}
		answered, _, err := runner.Generate(t.Context(), "", GenerateOptions{PromptTokenIDs: tokens, ProjectedInputs: &inputs, MaxNewTokens: 2})
		if err != nil || len(answered) <= len(tokens) {
			t.Fatalf("the text model gave no answer to the composite's prompt (%v)", err)
		}
		return prompt.Embeddings, want
	}
	red, redWant := rows(color.RGBA{R: 200, A: 255})
	for index := range red {
		if math.Abs(float64(red[index]-redWant[index])) > 1e-4 {
			t.Fatalf("row value %d is %v, the bridge's map gives %v", index, red[index], redWant[index])
		}
	}
	if blue, _ := rows(color.RGBA{B: 200, A: 255}); slices.Equal(blue, red) {
		t.Error("another picture gave the same rows")
	}
}

// TestACandidateCompositeServesItsPromotedRows holds a candidate composite, read before any recipe promotes it, to
// what its promotion would serve: for the same encoder, text model and map, its prompt holds the active composite's
// tokens, its rows at the same placeholders and the same rows; a map wider than the text model opens nothing.
func TestACandidateCompositeServesItsPromotedRows(t *testing.T) {
	t.Parallel()
	const placeholder = "</s>"
	runner := testcheck.Must(openFixtureRunnerWithResidency(testutil.HermeticLlamaGGUF(t, 64), OpenOptions{}, recipe.ResidencyHostReference))(t)
	t.Cleanup(func() { _ = runner.Close() })
	width := int(runner.spec.EmbeddingLength)
	first, bias := make([]float32, bridgeEncoderChannels*width), make([]float32, width)
	for index := range first {
		first[index] = float32(index%5) - 2
	}
	for index := range bias {
		bias[index] = float32(index) / 4
	}
	firstValue := testcheck.Must(reference.NewValue(tensor.MustShape(bridgeEncoderChannels, uint64(width)), first))(t)
	biasValue := testcheck.Must(reference.NewValue(tensor.MustShape(uint64(width)), bias))(t)
	weights := RepresentationBridgeWeights{First: &firstValue, FirstBias: &biasValue}
	store, authority := productionCompositionFixture(t, true, runner.ModelID(), uint64(width))
	source := &bridgeSourceFixture{model: authority.Recipe.SourceModel}
	target := bridgeTargetFixture{model: runner.ModelID(), width: width, placeholder: placeholder}
	promoted := testcheck.Must(OpenProductionComposition(t.Context(), store, source.model, runner.ModelID(), authority.Recipe.Task,
		&bridgeRuntimeResourcesFixture{source: source, target: target, weights: weights}))(t)
	candidate := testcheck.Must(OpenCandidateComposition(source, target, weights))(t)
	picture := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for index := range picture.Pix {
		picture.Pix[index] = uint8(index * 17)
	}
	want := testcheck.Must(promoted.BuildImagePrompt(t.Context(), runner, picture, "", "ab", false))(t)
	got := testcheck.Must(candidate.BuildImagePrompt(t.Context(), runner, picture, "", "ab", false))(t)
	if !slices.Equal(got.TokenIDs, want.TokenIDs) || !slices.Equal(got.EmbeddingTokenIndices, want.EmbeddingTokenIndices) || len(got.Embeddings) != len(want.Embeddings) {
		t.Fatalf("the candidate's prompt holds %d tokens and %d rows, the promoted composite's %d and %d", len(got.TokenIDs), len(got.EmbeddingTokenIndices), len(want.TokenIDs), len(want.EmbeddingTokenIndices))
	}
	for index, value := range want.Embeddings {
		if math.Abs(float64(got.Embeddings[index]-value)) > 1e-4*max(1, math.Abs(float64(value))) {
			t.Fatalf("row value %d is %v in the candidate, %v in the promoted composite", index, got.Embeddings[index], value)
		}
	}
	if _, err := OpenCandidateComposition(source, bridgeTargetFixture{model: runner.ModelID(), width: width - 1, placeholder: placeholder}, weights); err == nil {
		t.Error("a candidate whose map is wider than the text model opened")
	}
}

// productionCompositionFixture publishes an image composite's authority for a text model of width targetWidth: an
// encoder's output of bridgeEncoderChannels by bridgeEncoderTokens, a linear bridge with bias, both components on the
// host, and promotion evidence; activate makes the recipe the active one.
func productionCompositionFixture(t *testing.T, activate bool, targetModel artifact.ID, targetWidth uint64) (*overgodb.Store, composition.CompositionAuthority) {
	t.Helper()
	store := overgodbtest.Open(t, t.TempDir())
	ctx := t.Context()
	sourceModel := testutil.ArtifactID(t, artifact.KindModel, "bridge-source-model")
	_, sourceContract := inferenceBridgeContract(t, sourceModel, representation.TapEncoderOutput, representation.ModalityImage, bridgeEncoderChannels)
	_, targetContract := inferenceBridgeContract(t, targetModel, representation.TapEmbeddingOutput, representation.ModalityText, targetWidth)
	weights := testutil.ArtifactID(t, artifact.KindAdapter, "bridge-weights")
	inventory := testutil.ArtifactID(t, artifact.KindTensorInventory, "bridge-inventory")
	heldOut := testutil.ArtifactID(t, artifact.KindDatasetShard, "bridge-held-out")
	regression := testutil.ArtifactID(t, artifact.KindDatasetShard, "bridge-regression")
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "bridge-evaluator")
	trainingPolicy := testutil.ArtifactID(t, artifact.KindProfile, "bridge-training-policy")
	promotionPolicy := testcheck.Must(composition.NewRepresentationBridgePromotionPolicy(composition.RepresentationBridgePromotionPolicy{
		Direction: runrecord.DirectionMaximize, MinimumSeeds: 2,
		MinimumHeldOutGain: 0.05, MinimumSourceDependence: 0.1, MaximumRegression: 0.02,
		MaximumSeedSpread: 0.05, MaximumLatencyIncrease: 0.25, MaximumDeviceByteIncrease: 64,
	}))(t)
	parents := []artifact.ID{
		sourceModel, targetModel, sourceContract.Producer.Definition, targetContract.Producer.Definition,
		weights, inventory, heldOut, regression, evaluator, trainingPolicy,
	}
	descriptors := make([]artifact.Descriptor, 0, len(parents))
	for _, id := range parents {
		if _, found, err := store.Artifact(ctx, id); err != nil || found {
			testcheck.NoError(t, err)
			continue
		}
		descriptors = append(descriptors, artifact.Descriptor{ID: id, Size: 1})
	}
	testcheck.Must(store.Commit(ctx, artifact.Batch{Key: "fixture/production-composition/parents", Artifacts: descriptors}))(t)
	bridge := testcheck.Must(composition.NewBridgeDefinition(composition.BridgeDefinition{
		SourceModel: sourceModel, TargetModel: targetModel,
		Graph: bridgegraph.Definition{
			Source: sourceContract.ID, Target: targetContract.ID,
			Operator: bridgegraph.OperatorLinear, Bias: true,
		},
		Weights: weights, WeightInventory: inventory,
	}, sourceContract, targetContract))(t)
	execution := testcheck.Must((modelrecipe.RepresentationBridgeCompiler{
		SourcePlacement: recipe.PlacementHost, TargetPlacement: recipe.PlacementHost,
		SourceResidency: recipe.ResidencyHostReference, TargetResidency: recipe.ResidencyHostReference,
		SourceSession: recipe.SessionCapacity, TargetSession: recipe.SessionRequest,
	}).Definition(sourceModel, targetModel, sourceContract.ID, targetContract.ID, weights))(t)
	promotion := testcheck.Must((composition.RepresentationBridgePromoter{}).Evaluate(promotionPolicy, composition.RepresentationBridgePromotion{
		Bridge: weights, SourceModel: sourceModel, TargetModel: targetModel,
		SourceContract: sourceContract.ID, TargetContract: targetContract.ID,
		HeldOutSplit: heldOut, RegressionSet: regression, Evaluator: evaluator,
		Trials: []composition.RepresentationBridgePromotionTrial{
			{Seed: 11, BridgeScore: 0.9, CheapBaselineScore: 0.8, DroppedSourceScore: 0.7, ShuffledSourceScore: 0.72, RegressionBaselineScore: 0.85, RegressionCandidateScore: 0.84, BaselineLatencyNS: 100, ComposedLatencyNS: 110, BaselinePeakDeviceBytes: 1000, ComposedPeakDeviceBytes: 1020},
			{Seed: 29, BridgeScore: 0.88, CheapBaselineScore: 0.8, DroppedSourceScore: 0.69, ShuffledSourceScore: 0.7, RegressionBaselineScore: 0.86, RegressionCandidateScore: 0.85, BaselineLatencyNS: 100, ComposedLatencyNS: 112, BaselinePeakDeviceBytes: 1000, ComposedPeakDeviceBytes: 1024},
		},
	}))(t)
	compositionRecipe := testcheck.Must(composition.NewCompositionRecipe(composition.CompositionRecipe{
		SourceModel: sourceModel, TargetModel: targetModel, Task: recipe.TaskGeneration,
		SourceContract: sourceContract.ID, TargetContract: targetContract.ID,
		BridgeDefinition: bridge.ID, BridgeWeights: weights, ExecutionRecipe: execution.ID,
		TrainingPolicy: trainingPolicy, PromotionPolicy: promotionPolicy.ID, Promotion: promotion.ID,
	}))(t)
	authority := composition.CompositionAuthority{
		SourceContract: sourceContract, TargetContract: targetContract, Bridge: bridge,
		Execution: execution, PromotionPolicy: promotionPolicy, Promotion: promotion, Recipe: compositionRecipe,
	}
	testcheck.Must(store.Commit(ctx, testcheck.Must(authority.Batch("fixture/production-composition/authority"))(t)))(t)
	if activate {
		testcheck.Must(store.Commit(ctx, testcheck.Must(authority.Recipe.ActivationBatch(ctx, store, "fixture/production-composition/activate", nil))(t)))(t)
	}
	return store, authority
}

// inferenceBridgeContract is a contract for a model's representation at tap: width channels by up to four tokens.
func inferenceBridgeContract(
	t *testing.T,
	modelID artifact.ID,
	tap representation.TapPoint,
	modality representation.Modality,
	width uint64,
) ([]byte, representation.Contract) {
	t.Helper()
	value := representation.Contract{
		Version: representation.ContractVersion,
		Producer: representation.Producer{
			Model:      modelID,
			Definition: testutil.ArtifactID(t, artifact.KindModelDefinition, string(modality)+"-definition"),
			Tap:        tap,
		},
		Modality: modality,
		Tensor: representation.TensorContract{DataType: dtype.F32, Axes: []representation.Axis{
			{Kind: representation.AxisChannel, Bounds: representation.AxisBounds{Extent: width}},
			{Kind: representation.AxisSequence, Bounds: representation.AxisBounds{Minimum: 1, Maximum: bridgeEncoderTokens}},
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
	content := testcheck.Must(json.Marshal(value))(t)
	parsed := testcheck.Must(representation.ParseContract(content))(t)
	return content, parsed
}

func inferenceBridgeValue(t *testing.T, shape tensor.Shape, data []float32) reference.Value {
	t.Helper()
	return testcheck.Must(reference.NewValue(shape, data))(t)
}
