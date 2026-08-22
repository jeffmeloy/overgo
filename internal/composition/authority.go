package composition

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/recipe"
	"overgo/internal/representation"
)

const (
	// BridgeDefinitionVersion is the immutable bridge-definition document version.
	BridgeDefinitionVersion = artifact.InitialDocumentVersion
	// BridgeDefinitionMediaType identifies bridge-definition content in RepoDB.
	BridgeDefinitionMediaType = "application/vnd.overgo.representation-bridge-definition+json"
	// BridgeDefinitionSchema identifies the bridge-definition wire schema.
	BridgeDefinitionSchema = "overgo/representation-bridge-definition/v1"

	// CompositionRecipeVersion is the immutable composition-recipe document version.
	CompositionRecipeVersion = artifact.InitialDocumentVersion
	// CompositionRecipeMediaType identifies composition-recipe content in RepoDB.
	CompositionRecipeMediaType = "application/vnd.overgo.composition-recipe+json"
	// CompositionRecipeSchema identifies the composition-recipe wire schema.
	CompositionRecipeSchema = "overgo/composition-recipe/v1"
)

// BridgeDefinition binds one sealed operator and its immutable weights to the
// exact model and representation-contract identities it connects.
type BridgeDefinition struct {
	Version         uint16                 `json:"version"`
	SourceModel     artifact.ID            `json:"source_model"`
	TargetModel     artifact.ID            `json:"target_model"`
	Graph           bridgegraph.Definition `json:"graph"`
	Weights         artifact.ID            `json:"weights"`
	WeightInventory artifact.ID            `json:"weight_inventory"`
	ID              artifact.ID            `json:"-"`
}

// BridgeWeightAuthority is the exact immutable bridge payload and the tensor
// inventory that describes it.
type BridgeWeightAuthority struct {
	Weights   artifact.Descriptor
	Inventory artifact.Descriptor
}

// CompositionRecipe is the immutable activation subject. It joins the bridge
// authority to an executable projection recipe and exact promotion evidence.
type CompositionRecipe struct {
	Version                uint16      `json:"version"`
	SourceModel            artifact.ID `json:"source_model"`
	TargetModel            artifact.ID `json:"target_model"`
	Task                   recipe.Task `json:"task"`
	SourceContract         artifact.ID `json:"source_contract"`
	TargetContract         artifact.ID `json:"target_contract"`
	BridgeDefinition       artifact.ID `json:"bridge_definition"`
	BridgeWeights          artifact.ID `json:"bridge_weights"`
	ExecutionRecipe        artifact.ID `json:"execution_recipe"`
	TrainingPolicy         artifact.ID `json:"training_policy"`
	PromotionPolicy        artifact.ID `json:"promotion_policy"`
	Promotion              artifact.ID `json:"promotion"`
	ExternalCrossAttention artifact.ID `json:"external_cross_attention,omitzero"`
	ID                     artifact.ID `json:"-"`
}

// CompositionAuthority is the complete atomic publication set for one
// production candidate. External model, weight, dataset, and evaluator facts
// must already exist in the repository.
type CompositionAuthority struct {
	SourceContract         representation.Contract
	TargetContract         representation.Contract
	Bridge                 BridgeDefinition
	Execution              recipe.Definition
	PromotionPolicy        RepresentationBridgePromotionPolicy
	Promotion              RepresentationBridgePromotion
	ExternalCrossAttention *ExternalCrossAttentionDefinition
	Recipe                 CompositionRecipe
}

var bridgeDefinitionCodec = artifact.JSONDocumentCodec(
	"representation bridge definition", artifact.KindProfile,
	BridgeDefinitionMediaType, BridgeDefinitionSchema,
	canonicalizeBridgeDefinition,
	func(value BridgeDefinition) artifact.ID { return value.ID },
	func(value *BridgeDefinition, id artifact.ID) { value.ID = id },
	func(value BridgeDefinition) BridgeDefinition { return value },
)

var compositionRecipeCodec = artifact.JSONDocumentCodec(
	"composition recipe", artifact.KindRecipe,
	CompositionRecipeMediaType, CompositionRecipeSchema,
	canonicalizeCompositionRecipe,
	func(value CompositionRecipe) artifact.ID { return value.ID },
	func(value *CompositionRecipe, id artifact.ID) { value.ID = id },
	func(value CompositionRecipe) CompositionRecipe { return value },
)

// NewBridgeDefinition validates compatibility and identifies a bridge authority.
func NewBridgeDefinition(
	value BridgeDefinition,
	source, target representation.Contract,
) (BridgeDefinition, error) {
	value.Version, value.ID = BridgeDefinitionVersion, artifact.ID{}
	if value.SourceModel != source.Producer.Model || value.TargetModel != target.Producer.Model ||
		value.Graph.Source != source.ID || value.Graph.Target != target.ID {
		return BridgeDefinition{}, errors.New("composition: bridge definition differs from its representation authorities")
	}
	sourceContent, err := source.Content()
	if err != nil {
		return BridgeDefinition{}, err
	}
	targetContent, err := target.Content()
	if err != nil {
		return BridgeDefinition{}, err
	}
	if _, err := (bridgegraph.Compiler{}).Compile(value.Graph, sourceContent.Data, targetContent.Data); err != nil {
		return BridgeDefinition{}, fmt.Errorf("composition: bridge definition: %w", err)
	}
	return bridgeDefinitionCodec.New(value)
}

// ParseBridgeDefinition admits canonical bridge-definition bytes.
func ParseBridgeDefinition(data []byte) (BridgeDefinition, error) {
	return bridgeDefinitionCodec.Parse(data)
}

// Content returns the exact bridge-definition document.
func (value BridgeDefinition) Content() (artifact.Content, error) {
	return bridgeDefinitionCodec.Content(value)
}

// Lineage binds a bridge definition to models, contracts, weights, and operator assets.
func (value BridgeDefinition) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.SourceModel, value.TargetModel, value.Graph.Source, value.Graph.Target,
		value.Weights, value.WeightInventory,
	}
	for _, id := range []artifact.ID{value.Graph.Vocabulary, value.Graph.VocabularyHead, value.Graph.TargetEmbedding} {
		if id.Valid() {
			parents = append(parents, id)
		}
	}
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

// Batch prepares one bridge-definition publication.
func (value BridgeDefinition) Batch(key string) (artifact.Batch, error) {
	return bridgeDefinitionCodec.Batch(key, value, value.Lineage(), nil)
}

// NewCompositionRecipe validates and identifies one immutable activation subject.
func NewCompositionRecipe(value CompositionRecipe) (CompositionRecipe, error) {
	value.Version, value.ID = CompositionRecipeVersion, artifact.ID{}
	return compositionRecipeCodec.New(value)
}

// ParseCompositionRecipe admits canonical composition-recipe bytes.
func ParseCompositionRecipe(data []byte) (CompositionRecipe, error) {
	return compositionRecipeCodec.Parse(data)
}

// Content returns the exact composition-recipe document.
func (value CompositionRecipe) Content() (artifact.Content, error) {
	return compositionRecipeCodec.Content(value)
}

// Lineage binds a composition recipe to every immutable execution authority.
func (value CompositionRecipe) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, uniqueIDs([]artifact.ID{
		value.SourceModel, value.TargetModel, value.SourceContract, value.TargetContract,
		value.BridgeDefinition, value.BridgeWeights, value.ExecutionRecipe,
		value.TrainingPolicy, value.PromotionPolicy, value.Promotion,
		value.ExternalCrossAttention,
	})...)
}

// Batch prepares one composition-recipe publication.
func (value CompositionRecipe) Batch(key string) (artifact.Batch, error) {
	return compositionRecipeCodec.Batch(key, value, value.Lineage(), nil)
}

// Batch atomically publishes all composition documents without activating the
// recipe. External model, weight, dataset, and evaluator facts remain under
// their existing repository owners.
func (authority CompositionAuthority) Batch(key string) (artifact.Batch, error) {
	if err := validateCompositionAuthority(authority); err != nil {
		return artifact.Batch{}, err
	}
	var contents []artifact.Content
	lineage := make([]artifact.Lineage, 0)
	appendDocument := func(content artifact.Content, edges []artifact.Lineage, err error) error {
		if err != nil {
			return err
		}
		contents = append(contents, content)
		lineage = append(lineage, edges...)
		return nil
	}
	sourceContent, err := authority.SourceContract.Content()
	if err := appendDocument(sourceContent, authority.SourceContract.Lineage(), err); err != nil {
		return artifact.Batch{}, err
	}
	targetContent, err := authority.TargetContract.Content()
	if err := appendDocument(targetContent, authority.TargetContract.Lineage(), err); err != nil {
		return artifact.Batch{}, err
	}
	bridgeContent, err := authority.Bridge.Content()
	if err := appendDocument(bridgeContent, authority.Bridge.Lineage(), err); err != nil {
		return artifact.Batch{}, err
	}
	executionContent, err := authority.Execution.ArtifactContent()
	if err := appendDocument(executionContent, definitionLineage(authority.Execution), err); err != nil {
		return artifact.Batch{}, err
	}
	promotionContent, err := authority.Promotion.Content()
	policyContent, policyErr := authority.PromotionPolicy.Content()
	if err := appendDocument(policyContent, nil, policyErr); err != nil {
		return artifact.Batch{}, err
	}
	if err := appendDocument(promotionContent, authority.Promotion.Lineage(), err); err != nil {
		return artifact.Batch{}, err
	}
	if authority.ExternalCrossAttention != nil {
		externalContent, externalErr := authority.ExternalCrossAttention.Content()
		if err := appendDocument(externalContent, authority.ExternalCrossAttention.Lineage(), externalErr); err != nil {
			return artifact.Batch{}, err
		}
	}
	recipeContent, err := authority.Recipe.Content()
	if err := appendDocument(recipeContent, authority.Recipe.Lineage(), err); err != nil {
		return artifact.Batch{}, err
	}
	batch, err := artifact.NewDocumentBatch(key, contents, lineage, nil)
	if err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
}

// LoadBridgeDefinition resolves and revalidates one exact bridge authority.
func LoadBridgeDefinition(ctx context.Context, reader artifact.Reader, id artifact.ID) (BridgeDefinition, error) {
	content, err := loadCompositionContent(ctx, reader, id, artifact.KindProfile, BridgeDefinitionMediaType, BridgeDefinitionSchema)
	if err != nil {
		return BridgeDefinition{}, err
	}
	value, err := ParseBridgeDefinition(content.Data)
	if err != nil || value.ID != id {
		return BridgeDefinition{}, errors.Join(err, errors.New("composition: bridge definition identity differs"))
	}
	source, err := representation.LoadContract(ctx, reader, value.Graph.Source)
	if err != nil {
		return BridgeDefinition{}, err
	}
	target, err := representation.LoadContract(ctx, reader, value.Graph.Target)
	if err != nil {
		return BridgeDefinition{}, err
	}
	canonical, err := NewBridgeDefinition(value, source, target)
	if err != nil || canonical.ID != id {
		return BridgeDefinition{}, errors.Join(err, errors.New("composition: bridge definition authority differs"))
	}
	if _, err := loadBridgeWeightAuthority(ctx, reader, value); err != nil {
		return BridgeDefinition{}, err
	}
	return value, nil
}

// LoadBridgeWeights resolves the weight payload and inventory through their
// validated bridge definition instead of admitting caller-supplied identities.
func LoadBridgeWeights(
	ctx context.Context,
	reader artifact.Reader,
	bridgeDefinition artifact.ID,
) (BridgeWeightAuthority, error) {
	value, err := LoadBridgeDefinition(ctx, reader, bridgeDefinition)
	if err != nil {
		return BridgeWeightAuthority{}, err
	}
	return loadBridgeWeightAuthority(ctx, reader, value)
}

// LoadCompositionRecipe resolves a recipe and all exact authorities it names.
func LoadCompositionRecipe(ctx context.Context, reader artifact.Reader, id artifact.ID) (CompositionRecipe, error) {
	content, err := loadCompositionContent(ctx, reader, id, artifact.KindRecipe, CompositionRecipeMediaType, CompositionRecipeSchema)
	if err != nil {
		return CompositionRecipe{}, err
	}
	value, err := ParseCompositionRecipe(content.Data)
	if err != nil || value.ID != id {
		return CompositionRecipe{}, errors.Join(err, errors.New("composition: recipe identity differs"))
	}
	canonical, err := NewCompositionRecipe(value)
	if err != nil || canonical.ID != id {
		return CompositionRecipe{}, errors.Join(err, errors.New("composition: canonical recipe identity differs"))
	}
	authority, err := loadCompositionAuthority(ctx, reader, value)
	if err != nil {
		return CompositionRecipe{}, err
	}
	if err := validateCompositionAuthority(authority); err != nil {
		return CompositionRecipe{}, err
	}
	return value, nil
}

// ActivationBatch prepares one compare-and-set activation after resolving
// every immutable authority named by the recipe. RepoDB remains the sole
// mutation boundary.
func (value CompositionRecipe) ActivationBatch(
	ctx context.Context,
	reader artifact.Reader,
	key string,
	previous *artifact.ID,
) (artifact.Batch, error) {
	if ctx == nil || reader == nil {
		return artifact.Batch{}, errors.New("composition: activation repository authority is absent")
	}
	loaded, err := LoadCompositionRecipe(ctx, reader, value.ID)
	if err != nil || loaded != value {
		return artifact.Batch{}, errors.Join(err, errors.New("composition: activation recipe differs from repository authority"))
	}
	alias, err := ActiveCompositionAlias(value.SourceModel, value.TargetModel, value.Task)
	if err != nil {
		return artifact.Batch{}, err
	}
	current, active, err := ActiveComposition(ctx, reader, value.SourceModel, value.TargetModel, value.Task)
	if err != nil {
		return artifact.Batch{}, err
	}
	if previous == nil && active || previous != nil && (!active || current.ID != *previous) {
		return artifact.Batch{}, errors.New("composition: activation compare-and-set authority differs")
	}
	batch := artifact.Batch{
		Key:     key,
		Aliases: []artifact.AliasBinding{{Name: alias, Target: value.ID, Previous: previous}},
	}
	if err := batch.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
}

// ActiveCompositionAlias derives the source-target-task scoped alias name.
func ActiveCompositionAlias(source, target artifact.ID, task recipe.Task) (string, error) {
	if source.Kind() != artifact.KindModel || target.Kind() != artifact.KindModel || source == target || !task.Valid() {
		return "", errors.New("composition: active alias scope is invalid")
	}
	return "composition.active." + string(task) + "." + source.String() + "." + target.String(), nil
}

// ActiveComposition resolves and revalidates the exact scoped active recipe.
func ActiveComposition(
	ctx context.Context,
	reader artifact.Reader,
	source, target artifact.ID,
	task recipe.Task,
) (CompositionRecipe, bool, error) {
	alias, err := ActiveCompositionAlias(source, target, task)
	if err != nil {
		return CompositionRecipe{}, false, err
	}
	id, found, err := artifact.ResolveAlias(ctx, reader, alias)
	if err != nil || !found {
		return CompositionRecipe{}, found, err
	}
	value, err := LoadCompositionRecipe(ctx, reader, id)
	if err != nil {
		return CompositionRecipe{}, false, err
	}
	if value.SourceModel != source || value.TargetModel != target || value.Task != task {
		return CompositionRecipe{}, false, errors.New("composition: active alias subject mismatch")
	}
	return value, true, nil
}

func canonicalizeBridgeDefinition(value *BridgeDefinition) error {
	if value == nil || value.Version != BridgeDefinitionVersion ||
		value.SourceModel.Kind() != artifact.KindModel || value.TargetModel.Kind() != artifact.KindModel ||
		value.SourceModel == value.TargetModel || value.Graph.Source.Kind() != artifact.KindProfile ||
		value.Graph.Target.Kind() != artifact.KindProfile || value.Graph.Source == value.Graph.Target ||
		!value.Graph.Operator.Valid() || value.Weights.Kind() != artifact.KindAdapter ||
		value.WeightInventory.Kind() != artifact.KindTensorInventory {
		return errors.New("composition: invalid bridge definition authority")
	}
	return nil
}

func canonicalizeCompositionRecipe(value *CompositionRecipe) error {
	if value == nil || value.Version != CompositionRecipeVersion ||
		value.SourceModel.Kind() != artifact.KindModel || value.TargetModel.Kind() != artifact.KindModel ||
		value.SourceModel == value.TargetModel || !value.Task.Valid() ||
		value.SourceContract.Kind() != artifact.KindProfile || value.TargetContract.Kind() != artifact.KindProfile ||
		value.BridgeDefinition.Kind() != artifact.KindProfile || value.BridgeWeights.Kind() != artifact.KindAdapter ||
		value.ExecutionRecipe.Kind() != artifact.KindRecipe || value.TrainingPolicy.Kind() != artifact.KindProfile ||
		value.PromotionPolicy.Kind() != artifact.KindProfile || value.Promotion.Kind() != artifact.KindEvidence ||
		value.ExternalCrossAttention.Valid() && value.ExternalCrossAttention.Kind() != artifact.KindProfile {
		return errors.New("composition: invalid composition recipe authority")
	}
	return nil
}

func validateCompositionAuthority(value CompositionAuthority) error {
	if err := value.SourceContract.ValidateIdentity(); err != nil {
		return err
	}
	if err := value.TargetContract.ValidateIdentity(); err != nil {
		return err
	}
	bridge, err := NewBridgeDefinition(value.Bridge, value.SourceContract, value.TargetContract)
	if err != nil || bridge.ID != value.Bridge.ID {
		return errors.Join(err, errors.New("composition: bridge authority identity differs"))
	}
	if err := value.Promotion.ValidateIdentity(); err != nil {
		return err
	}
	if err := value.PromotionPolicy.ValidateIdentity(); err != nil {
		return err
	}
	sourceChannels, hasSourceChannels := representationAxisExtent(value.SourceContract.Tensor, representation.AxisChannel)
	if value.Recipe.ExternalCrossAttention.Valid() {
		if value.ExternalCrossAttention == nil ||
			value.ExternalCrossAttention.ValidateIdentity() != nil ||
			value.ExternalCrossAttention.ID != value.Recipe.ExternalCrossAttention ||
			value.ExternalCrossAttention.Source != value.Recipe.SourceModel ||
			value.ExternalCrossAttention.Target != value.Recipe.TargetModel ||
			value.ExternalCrossAttention.Adapter != value.Recipe.BridgeWeights ||
			!hasSourceChannels || value.ExternalCrossAttention.SourceChannels != sourceChannels {
			return errors.New("composition: external cross-attention authority differs")
		}
	} else if value.ExternalCrossAttention != nil {
		return errors.New("composition: unreferenced external cross-attention authority")
	}
	if err := value.Execution.ValidateIdentity(); err != nil {
		return err
	}
	canonicalRecipe, err := NewCompositionRecipe(value.Recipe)
	if err != nil || canonicalRecipe.ID != value.Recipe.ID {
		return errors.Join(err, errors.New("composition: recipe authority identity differs"))
	}
	if value.Recipe.SourceModel != value.SourceContract.Producer.Model ||
		value.Recipe.TargetModel != value.TargetContract.Producer.Model ||
		value.Recipe.SourceContract != value.SourceContract.ID ||
		value.Recipe.TargetContract != value.TargetContract.ID ||
		value.Recipe.BridgeDefinition != value.Bridge.ID ||
		value.Recipe.BridgeWeights != value.Bridge.Weights ||
		value.Recipe.ExecutionRecipe != value.Execution.ID ||
		value.Recipe.PromotionPolicy != value.PromotionPolicy.ID ||
		value.Recipe.Promotion != value.Promotion.ID ||
		value.Execution.Task != recipe.TaskProjection ||
		value.Promotion.Bridge != value.Bridge.Weights ||
		value.Promotion.PolicyID != value.PromotionPolicy.ID ||
		value.Promotion.Policy != value.PromotionPolicy ||
		value.Promotion.SourceModel != value.Recipe.SourceModel ||
		value.Promotion.TargetModel != value.Recipe.TargetModel ||
		value.Promotion.SourceContract != value.SourceContract.ID ||
		value.Promotion.TargetContract != value.TargetContract.ID {
		return errors.New("composition: composition authority cross-reference differs")
	}
	return nil
}

func representationAxisExtent(value representation.TensorContract, kind representation.AxisKind) (uint64, bool) {
	for _, axis := range value.Axes {
		if axis.Kind == kind {
			return axis.Bounds.Extent, true
		}
	}
	var extent uint64
	return extent, false
}

func loadCompositionAuthority(
	ctx context.Context,
	reader artifact.Reader,
	value CompositionRecipe,
) (CompositionAuthority, error) {
	source, err := representation.LoadContract(ctx, reader, value.SourceContract)
	if err != nil {
		return CompositionAuthority{}, err
	}
	target, err := representation.LoadContract(ctx, reader, value.TargetContract)
	if err != nil {
		return CompositionAuthority{}, err
	}
	bridge, err := LoadBridgeDefinition(ctx, reader, value.BridgeDefinition)
	if err != nil {
		return CompositionAuthority{}, err
	}
	if _, err := LoadBridgeWeights(ctx, reader, value.BridgeDefinition); err != nil {
		return CompositionAuthority{}, err
	}
	if _, found, err := reader.Artifact(ctx, value.TrainingPolicy); err != nil || !found {
		return CompositionAuthority{}, errors.Join(err, errors.New("composition: training policy is absent"))
	}
	promotionPolicy, err := LoadRepresentationBridgePromotionPolicy(ctx, reader, value.PromotionPolicy)
	if err != nil {
		return CompositionAuthority{}, err
	}
	promotion, err := LoadRepresentationBridgePromotion(ctx, reader, value.Promotion)
	if err != nil {
		return CompositionAuthority{}, err
	}
	if promotionPolicy != promotion.Policy {
		return CompositionAuthority{}, errors.New("composition: promotion policy differs from promotion evidence")
	}
	var external *ExternalCrossAttentionDefinition
	if value.ExternalCrossAttention.Valid() {
		loaded, loadErr := LoadExternalCrossAttentionDefinition(ctx, reader, value.ExternalCrossAttention)
		if loadErr != nil {
			return CompositionAuthority{}, loadErr
		}
		external = &loaded
	}
	executionContent, err := loadCompositionContent(ctx, reader, value.ExecutionRecipe, artifact.KindRecipe, recipe.MediaType, recipe.Schema)
	if err != nil {
		return CompositionAuthority{}, err
	}
	execution, err := recipe.ParseDefinition(executionContent.Data)
	if err != nil || execution.ID != value.ExecutionRecipe {
		return CompositionAuthority{}, errors.Join(err, errors.New("composition: execution recipe identity differs"))
	}
	return CompositionAuthority{
		SourceContract: source, TargetContract: target, Bridge: bridge,
		Execution: execution, PromotionPolicy: promotionPolicy, Promotion: promotion,
		ExternalCrossAttention: external, Recipe: value,
	}, nil
}

func loadCompositionContent(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	kind artifact.Kind,
	mediaType, schema string,
) (artifact.Content, error) {
	if ctx == nil || reader == nil || id.Kind() != kind {
		return artifact.Content{}, errors.New("composition: repository query authority is invalid")
	}
	content, found, err := reader.Content(ctx, id)
	if err != nil {
		return artifact.Content{}, err
	}
	if !found || content.Descriptor.ID != id || content.Descriptor.MediaType != mediaType ||
		content.Descriptor.Schema != schema || content.Validate() != nil {
		return artifact.Content{}, errors.New("composition: repository content is absent or incompatible")
	}
	return content, nil
}

func loadBridgeWeightAuthority(
	ctx context.Context,
	reader artifact.Reader,
	value BridgeDefinition,
) (BridgeWeightAuthority, error) {
	weights, found, err := reader.Artifact(ctx, value.Weights)
	if err != nil || !found {
		return BridgeWeightAuthority{}, errors.Join(err, errors.New("composition: bridge weights are absent"))
	}
	inventory, found, err := reader.Artifact(ctx, value.WeightInventory)
	if err != nil || !found {
		return BridgeWeightAuthority{}, errors.Join(err, errors.New("composition: bridge weight inventory is absent"))
	}
	if weights.ID != value.Weights || inventory.ID != value.WeightInventory {
		return BridgeWeightAuthority{}, errors.New("composition: bridge weight authority identity differs")
	}
	return BridgeWeightAuthority{Weights: weights, Inventory: inventory}, nil
}

func definitionLineage(value recipe.Definition) []artifact.Lineage {
	parents := make([]artifact.ID, len(value.Dependencies))
	for index, dependency := range value.Dependencies {
		parents[index] = dependency.Artifact
	}
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

func uniqueIDs(values []artifact.ID) []artifact.ID {
	seen := make(map[artifact.ID]struct{}, len(values))
	result := make([]artifact.ID, 0, len(values))
	for _, value := range values {
		if !value.Valid() {
			continue
		}
		if _, duplicate := seen[value]; duplicate {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}
