package composition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipe"
)

const (
	// CompositeGenerationPromotionVersion is the immutable surface-promotion version.
	CompositeGenerationPromotionVersion = artifact.InitialDocumentVersion
	// CompositeGenerationPromotionMediaType identifies production generation promotion evidence.
	CompositeGenerationPromotionMediaType = "application/vnd.overgo.composite-generation-promotion+json"
	// CompositeGenerationPromotionSchema identifies the promotion wire schema.
	CompositeGenerationPromotionSchema = "overgo/composite-generation-promotion/v1"
)

// CompositeGenerationPromotion is the sole generation-surface authority. It
// joins the current active plan to real CUDA parity evidence and one exact
// generated output that runtime, API, and GUI must expose without substitution.
type CompositeGenerationPromotion struct {
	Version           uint16      `json:"version"`
	SourceModel       artifact.ID `json:"source_model"`
	TargetModel       artifact.ID `json:"target_model"`
	Task              recipe.Task `json:"task"`
	CompositionRecipe artifact.ID `json:"composition_recipe"`
	ExecutionPlan     artifact.ID `json:"execution_plan"`
	BridgePromotion   artifact.ID `json:"bridge_promotion"`
	CUDAEvidence      artifact.ID `json:"cuda_evidence"`
	Output            artifact.ID `json:"output"`
	ID                artifact.ID `json:"-"`
}

// CompositeGenerationPromotionAuthority validates surface promotion but does
// not activate composition recipes or execute either model.
type CompositeGenerationPromotionAuthority struct{}

var compositeGenerationPromotionCodec = artifact.JSONDocumentCodec(
	"composite generation promotion", artifact.KindEvidence,
	CompositeGenerationPromotionMediaType, CompositeGenerationPromotionSchema,
	canonicalizeCompositeGenerationPromotion,
	func(value CompositeGenerationPromotion) artifact.ID { return value.ID },
	func(value *CompositeGenerationPromotion, id artifact.ID) { value.ID = id },
	func(value CompositeGenerationPromotion) CompositeGenerationPromotion { return value },
)

// Promote admits only CUDA evidence for the exact current active plan.
func (CompositeGenerationPromotionAuthority) Promote(
	plan CompositionExecutionPlan,
	evidence CompositeGenerationCUDAEvidence,
) (CompositeGenerationPromotion, error) {
	if plan.ValidateIdentity() != nil || evidence.ValidateIdentity() != nil ||
		evidence.SourceModel != plan.SourceModel || evidence.TargetModel != plan.TargetModel ||
		evidence.CompositionRecipe != plan.CompositionRecipe || evidence.ExecutionPlan != plan.ID ||
		evidence.Promotion != plan.Promotion || evidence.ComposedOutput != evidence.BaselineOutput ||
		!evidence.ExactOutputParity {
		return CompositeGenerationPromotion{}, errors.New("composition: composite generation CUDA evidence does not promote the active plan")
	}
	return compositeGenerationPromotionCodec.New(CompositeGenerationPromotion{
		Version:     CompositeGenerationPromotionVersion,
		SourceModel: plan.SourceModel, TargetModel: plan.TargetModel, Task: plan.Task,
		CompositionRecipe: plan.CompositionRecipe, ExecutionPlan: plan.ID,
		BridgePromotion: plan.Promotion, CUDAEvidence: evidence.ID, Output: evidence.ComposedOutput,
	})
}

// Parse admits canonical serialized surface-promotion evidence.
func (CompositeGenerationPromotionAuthority) Parse(data []byte) (CompositeGenerationPromotion, error) {
	return compositeGenerationPromotionCodec.Parse(data)
}

// Load requires exact surface-promotion content from OvergoDB.
func (CompositeGenerationPromotionAuthority) Load(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (CompositeGenerationPromotion, error) {
	return compositeGenerationPromotionCodec.Require(ctx, reader, id)
}

// ValidateIdentity verifies the promotion envelope and content identity.
func (value CompositeGenerationPromotion) ValidateIdentity() error {
	return compositeGenerationPromotionCodec.ValidateIdentity(value)
}

// Content returns exact surface-promotion content.
func (value CompositeGenerationPromotion) Content() (artifact.Content, error) {
	return compositeGenerationPromotionCodec.Content(value)
}

// Lineage binds promotion to the active recipe, plan, bridge promotion, CUDA
// evidence, source, target, and generated output.
func (value CompositeGenerationPromotion) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(
		value.ID, value.SourceModel, value.TargetModel, value.CompositionRecipe,
		value.ExecutionPlan, value.BridgePromotion, value.CUDAEvidence, value.Output,
	)
}

// CompositeGenerationPromotionAlias scopes production generation by source,
// target, and task, exactly like the active composition recipe.
func CompositeGenerationPromotionAlias(source, target artifact.ID, task recipe.Task) (string, error) {
	if source.Kind() != artifact.KindModel || target.Kind() != artifact.KindModel || source == target || !task.Valid() {
		return "", errors.New("composition: composite generation promotion scope is invalid")
	}
	return "composition.generation.active." + string(task) + "." + source.String() + "." + target.String(), nil
}

// ActiveCompositeGeneration resolves the promoted generation and revalidates
// it against the current active execution plan. Stale or unpromoted aliases
// are refused rather than falling back to caller-supplied construction.
func ActiveCompositeGeneration(
	ctx context.Context,
	reader artifact.Reader,
	source, target artifact.ID,
	task recipe.Task,
) (CompositeGenerationPromotion, CompositionExecutionPlan, bool, error) {
	alias, err := CompositeGenerationPromotionAlias(source, target, task)
	if err != nil {
		return CompositeGenerationPromotion{}, CompositionExecutionPlan{}, false, err
	}
	promotion, found, err := compositeGenerationPromotionCodec.Resolve(ctx, reader, alias)
	if err != nil || !found {
		return CompositeGenerationPromotion{}, CompositionExecutionPlan{}, found, err
	}
	plan, err := CompileCompositionExecutionPlan(ctx, reader, source, target, task)
	if err != nil {
		return CompositeGenerationPromotion{}, CompositionExecutionPlan{}, false, err
	}
	if promotion.SourceModel != source || promotion.TargetModel != target || promotion.Task != task ||
		promotion.CompositionRecipe != plan.CompositionRecipe || promotion.ExecutionPlan != plan.ID ||
		promotion.BridgePromotion != plan.Promotion {
		return CompositeGenerationPromotion{}, CompositionExecutionPlan{}, false,
			errors.New("composition: promoted generation differs from the active recipe")
	}
	cudaEvidence, err := (CompositeGenerationCUDAAuthority{}).Load(ctx, reader, promotion.CUDAEvidence)
	if err != nil {
		return CompositeGenerationPromotion{}, CompositionExecutionPlan{}, false, err
	}
	expected, err := (CompositeGenerationPromotionAuthority{}).Promote(plan, cudaEvidence)
	if err != nil || expected.ID != promotion.ID {
		return CompositeGenerationPromotion{}, CompositionExecutionPlan{}, false,
			errors.Join(err, errors.New("composition: promoted generation evidence is stale"))
	}
	return promotion, plan, true, nil
}

func canonicalizeCompositeGenerationPromotion(value *CompositeGenerationPromotion) error {
	if value == nil || value.Version != CompositeGenerationPromotionVersion ||
		value.SourceModel.Kind() != artifact.KindModel || value.TargetModel.Kind() != artifact.KindModel ||
		value.SourceModel == value.TargetModel || !value.Task.Valid() ||
		value.CompositionRecipe.Kind() != artifact.KindRecipe || value.ExecutionPlan.Kind() != artifact.KindProfile ||
		value.BridgePromotion.Kind() != artifact.KindEvidence || value.CUDAEvidence.Kind() != artifact.KindEvidence ||
		value.Output.Kind() != artifact.KindOutput || !checked.Nonzero(value.Output) {
		return errors.New("composition: invalid composite generation promotion")
	}
	return nil
}
