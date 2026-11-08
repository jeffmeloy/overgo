package main

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/checked"
	"overgo/internal/composition"
	"overgo/internal/jsonfile"
	"overgo/internal/mediacapability"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

const (
	// adapterMediaType is how the store holds a bridge's adapter: the safetensors file of the map and its bias the
	// caption-projector route writes.
	adapterMediaType = "application/vnd.safetensors"
	// candidateKey begins the key a composite's publication commits under, its recipe's identity ending it.
	candidateKey = "composition/candidate/"
)

// publishSpec names a judged image composite: the encoder directory, the text model's file and the bridge's adapter
// between them, the tokens the encoder gives a picture, the files its judgement read and kept (the held-out split and
// the regression set, the envelope's record as the evaluator and the run's record as the training policy), and the
// promotion policy with the trials it judges.
type publishSpec struct {
	Encoder        string                                           `json:"encoder"`
	TextModel      string                                           `json:"text_model"`
	Adapter        string                                           `json:"adapter"`
	EncoderTokens  uint64                                           `json:"encoder_tokens"`
	HeldOutSplit   string                                           `json:"held_out_split"`
	RegressionSet  string                                           `json:"regression_set"`
	Evaluator      string                                           `json:"evaluator"`
	TrainingPolicy string                                           `json:"training_policy"`
	Policy         composition.RepresentationBridgePromotionPolicy  `json:"policy"`
	Trials         []composition.RepresentationBridgePromotionTrial `json:"trials"`
}

// compositeCandidate is what a publication commits: the composite's authority, the files it cites with the edge from
// the adapter's inventory to the adapter, and what the store must learn of the two models before it may cite them: the
// encoder's image-embedding recipe, and the text model's derivation when no recipe of the store's names it.
type compositeCandidate struct {
	authority  composition.CompositionAuthority
	files      []artifact.Content
	lineage    []artifact.Lineage
	encoder    recipe.Definition
	derivation *modelrecipe.Derivation
}

// publishComposite publishes a judged image composite to the store at repository as an inactive candidate, the recipe
// cmd/evaluate serves once its owner activates it: the contracts of the encoder's output and of the text model's
// embeddings, the affine bridge over the adapter, the execution recipe placing both on the card, the promotion the
// trials earn under the policy, and the composition recipe joining them, in one commit beside the cited files the store
// lacks. Trials the policy refuses commit nothing; a candidate the store holds is not committed again.
func publishComposite(repository, specPath string) (composition.CompositionAuthority, error) {
	var spec publishSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return composition.CompositionAuthority{}, err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return composition.CompositionAuthority{}, err
	}
	defer store.Close()
	ctx := context.Background()
	candidate, err := spec.candidate(ctx, store)
	if err != nil {
		return composition.CompositionAuthority{}, err
	}
	authority := candidate.authority
	batch, err := authority.Batch(candidateKey + authority.Recipe.ID.String())
	if err != nil {
		return composition.CompositionAuthority{}, err
	}
	if _, published, err := store.Artifact(ctx, authority.Recipe.ID); err != nil || published {
		return authority, err
	}
	// Only a promotion earned publishes the models, each as its registration publishes it.
	if err := candidate.register(ctx, store, spec.Encoder); err != nil {
		return composition.CompositionAuthority{}, err
	}
	for _, file := range candidate.files {
		held, err := store.HasContent(ctx, file.Descriptor.ID)
		if err != nil {
			return composition.CompositionAuthority{}, err
		}
		if !held {
			batch.Contents = append(batch.Contents, file)
		}
	}
	batch.Lineage = append(batch.Lineage, candidate.lineage...)
	if _, err := store.Commit(ctx, batch); err != nil {
		return composition.CompositionAuthority{}, err
	}
	return authority, nil
}

// candidate builds the composite's authority from the spec without writing to the store: the models as the serving
// path identifies them, the encoder as cmd/evaluate resolves an image encoder and the text model as its runner does,
// and the promotion as the policy judges the trials.
func (spec publishSpec) candidate(ctx context.Context, store *overgodb.Store) (compositeCandidate, error) {
	var candidate compositeCandidate
	var adapter, heldOut, regression, evaluator, training artifact.ID
	for _, cited := range []struct {
		path      string
		kind      artifact.Kind
		mediaType string
		id        *artifact.ID
	}{
		{spec.Adapter, artifact.KindAdapter, adapterMediaType, &adapter},
		{spec.HeldOutSplit, artifact.KindDatasetShard, artifact.JSONMediaType, &heldOut},
		{spec.RegressionSet, artifact.KindDatasetShard, artifact.JSONMediaType, &regression},
		{spec.Evaluator, artifact.KindEvidence, artifact.JSONMediaType, &evaluator},
		{spec.TrainingPolicy, artifact.KindProfile, artifact.JSONMediaType, &training},
	} {
		data, err := artifact.ReadContentFile(cited.path)
		if err == nil {
			*cited.id, err = artifact.IdentifyBytes(cited.kind, data)
		}
		if err != nil {
			return compositeCandidate{}, fmt.Errorf("seam-align: %s: %w", cited.path, err)
		}
		candidate.files = append(candidate.files, artifact.Content{
			Descriptor: artifact.Descriptor{ID: *cited.id, Size: uint64(len(data)), MediaType: cited.mediaType}, Data: data,
		})
	}
	bridge, err := rae.LoadEncoderBridge(spec.Adapter)
	if err != nil {
		return compositeCandidate{}, err
	}
	textWidth := uint64(len(bridge.Bias))
	encoderWidth, affine := checked.DivExact64(uint64(len(bridge.Weight)), textWidth)
	if !affine {
		return compositeCandidate{}, fmt.Errorf("seam-align: %s holds no map of the encoder's channels for each of the text model's", spec.Adapter)
	}
	weights, err := safetensors.OpenFile(spec.Adapter)
	if err != nil {
		return compositeCandidate{}, err
	}
	inventory, err := modelartifact.NewSafetensorsTensorInventory(adapter, weights)
	if err = errors.Join(err, weights.Close()); err != nil {
		return compositeCandidate{}, err
	}
	inventoryContent, err := inventory.Content()
	if err != nil {
		return compositeCandidate{}, err
	}
	candidate.files, candidate.lineage = append(candidate.files, inventoryContent), inventory.Lineage()
	source, err := mediacapability.Catalog[recipe.TaskImageEmbedding].Resolve(ctx, spec.Encoder)
	if err != nil {
		return compositeCandidate{}, err
	}
	sourceModel := source.Inventory.Manifest.ID
	// The encoder's definition binds it to its image-embedding recipe, the active one where one stands, as cmd/evaluate
	// records an image encoder's.
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, sourceModel, recipe.TaskImageEmbedding)
	candidate.encoder = activation.Definition
	if err == nil && !active {
		candidate.encoder, err = modelrecipe.CapabilityDefinition(recipe.TaskImageEmbedding, sourceModel)
	}
	if err != nil {
		return compositeCandidate{}, err
	}
	sourceDefinition, err := modelrecipe.TaskModelDefinitionID(candidate.encoder)
	if err != nil {
		return compositeCandidate{}, err
	}
	text, derivation, width, err := textModel(ctx, store, spec.TextModel)
	if err != nil {
		return compositeCandidate{}, err
	}
	if width != textWidth {
		return compositeCandidate{}, fmt.Errorf("seam-align: the bridge gives %d channels and %s embeds in %d", textWidth, spec.TextModel, width)
	}
	candidate.derivation = derivation
	// Each side of the seam is the representation its model gives at its tap, its width by the tokens the encoder gives
	// a picture, every one valid and in order, at the model's own scale.
	var sourceContract, targetContract representation.Contract
	for _, side := range []struct {
		model, definition artifact.ID
		tap               representation.TapPoint
		modality          representation.Modality
		width             uint64
		contract          *representation.Contract
	}{
		{sourceModel, sourceDefinition, representation.TapEncoderOutput, representation.ModalityImage, encoderWidth, &sourceContract},
		{text.Model, text.Definition, representation.TapEmbeddingOutput, representation.ModalityText, textWidth, &targetContract},
	} {
		*side.contract, err = representation.NewContract(representation.Contract{
			Producer: representation.Producer{Model: side.model, Definition: side.definition, Tap: side.tap},
			Modality: side.modality,
			Tensor: representation.TensorContract{DataType: dtype.F32, Axes: []representation.Axis{
				{Kind: representation.AxisChannel, Bounds: representation.AxisBounds{Extent: side.width}},
				{Kind: representation.AxisSequence, Bounds: representation.AxisBounds{Extent: spec.EncoderTokens}},
			}},
			Sequence: representation.SequenceContract{
				Axis: representation.AxisSequence, Mask: representation.MaskNone, Padding: representation.PaddingNone,
				Position: representation.PositionSequential, PositionAxes: []representation.AxisKind{representation.AxisSequence},
			},
			Normalization: representation.NormalizationContract{
				Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative,
			},
		})
		if err != nil {
			return compositeCandidate{}, err
		}
	}
	bridgeDefinition, err := composition.NewBridgeDefinition(composition.BridgeDefinition{
		SourceModel: sourceModel, TargetModel: text.Model,
		Graph: bridgegraph.Definition{
			Source: sourceContract.ID, Target: targetContract.ID, Operator: bridgegraph.OperatorLinear, Bias: true,
		},
		Weights: adapter, WeightInventory: inventory.ID,
	}, sourceContract, targetContract)
	if err != nil {
		return compositeCandidate{}, err
	}
	// Both run on the card: the encoder with its every weight resident there in F32 (vitencoder.Card), open across
	// pictures, and the text model with the native stores its runner preloads, a request's.
	execution, err := (modelrecipe.RepresentationBridgeCompiler{
		SourcePlacement: recipe.PlacementDevice, TargetPlacement: recipe.PlacementDevice,
		SourceResidency: recipe.ResidencyDeviceF32, TargetResidency: recipe.ResidencyDeviceNative,
		SourceSession: recipe.SessionCapacity, TargetSession: recipe.SessionRequest,
	}).Definition(sourceModel, text.Model, sourceContract.ID, targetContract.ID, adapter)
	if err != nil {
		return compositeCandidate{}, err
	}
	policy, err := composition.NewRepresentationBridgePromotionPolicy(spec.Policy)
	if err != nil {
		return compositeCandidate{}, err
	}
	promotion, err := (composition.RepresentationBridgePromoter{}).Evaluate(policy, composition.RepresentationBridgePromotion{
		Bridge: adapter, SourceModel: sourceModel, TargetModel: text.Model,
		SourceContract: sourceContract.ID, TargetContract: targetContract.ID,
		HeldOutSplit: heldOut, RegressionSet: regression, Evaluator: evaluator, Trials: spec.Trials,
	})
	if err != nil {
		return compositeCandidate{}, err
	}
	// The task is the one cmd/evaluate serves a composite under.
	compositionRecipe, err := composition.NewCompositionRecipe(composition.CompositionRecipe{
		SourceModel: sourceModel, TargetModel: text.Model, Task: recipe.TaskGeneration,
		SourceContract: sourceContract.ID, TargetContract: targetContract.ID,
		BridgeDefinition: bridgeDefinition.ID, BridgeWeights: adapter, ExecutionRecipe: execution.ID,
		TrainingPolicy: training, PromotionPolicy: policy.ID, Promotion: promotion.ID,
	})
	candidate.authority = composition.CompositionAuthority{
		SourceContract: sourceContract, TargetContract: targetContract, Bridge: bridgeDefinition,
		Execution: execution, PromotionPolicy: policy, Promotion: promotion, Recipe: compositionRecipe,
	}
	return candidate, err
}

// register publishes what the store must hold of the two models before the composite cites them, as registering each
// publishes it: the encoder's facts, its image-embedding recipe as a candidate where the store holds none, and the
// definition binding that recipe to it; and the text model's facts and recipe as its first open publishes them, where
// no recipe of the store's names it.
func (candidate compositeCandidate) register(ctx context.Context, store *overgodb.Store, encoderPath string) error {
	capability := mediacapability.Catalog[recipe.TaskImageEmbedding]
	if _, _, err := mediacapability.Prepare(ctx, store, encoderPath, recipe.TaskImageEmbedding, capability); err != nil {
		return err
	}
	_, published, err := modelrecipe.Status(ctx, store, candidate.encoder.ID)
	if err == nil && !published {
		_, _, err = modelrecipe.PublishCandidate(ctx, store, "recipe/candidate/"+candidate.encoder.ID.String(), candidate.encoder)
	}
	if err == nil {
		_, err = modelrecipe.PublishTaskModelDefinition(ctx, store, candidate.encoder.ID)
	}
	if err == nil && candidate.derivation != nil {
		err = modelrecipe.PublishDerivation(ctx, store, *candidate.derivation)
	}
	return err
}

// textModel resolves the text model at path as its runner resolves it, through the store's claimed load: the identity
// the runner reports, the derivation the store must learn when no recipe of the store's names the model, and its
// embedding width.
func textModel(ctx context.Context, store *overgodb.Store, path string) (modelrecipe.ProgramIdentity, *modelrecipe.Derivation, uint64, error) {
	// The model is read from its bytes, never from a memo's claim that its size and time kept (SQA-331).
	loaded, err := modelrecipe.ResolveActiveGGUF(ctx, store, path)
	if err != nil {
		return modelrecipe.ProgramIdentity{}, nil, 0, err
	}
	identity, err := loaded.Identity()
	var derivation *modelrecipe.Derivation
	if derived, found := loaded.Derived(); found {
		derivation = &derived
	}
	file, _, spec, _, _, _, takeErr := loaded.Take()
	if takeErr == nil {
		takeErr = file.Close()
	}
	return identity, derivation, uint64(spec.EmbeddingLength), errors.Join(err, takeErr)
}
