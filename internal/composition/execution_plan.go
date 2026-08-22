package composition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/checked"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/tensor"
)

const (
	// CompositionExecutionPlanVersion is the immutable compiled-plan version.
	CompositionExecutionPlanVersion = artifact.InitialDocumentVersion
	// CompositionExecutionPlanMediaType identifies compiled composition plans.
	CompositionExecutionPlanMediaType = "application/vnd.overgo.composition-execution-plan+json"
	// CompositionExecutionPlanSchema identifies the compiled-plan wire schema.
	CompositionExecutionPlanSchema = "overgo/composition-execution-plan/v1"
)

// CompositionBoundary fixes one capture or injection seam to an exact model,
// contract, tap, optional layer, and channel extent.
type CompositionBoundary struct {
	Contract artifact.ID             `json:"contract"`
	Model    artifact.ID             `json:"model"`
	Tap      representation.TapPoint `json:"tap"`
	Layer    *uint32                 `json:"layer,omitempty"`
	Channels uint64                  `json:"channels"`
}

// CompositionExecutionPlan is the sole typed runtime authority compiled from
// an active composition recipe and every immutable authority it references.
type CompositionExecutionPlan struct {
	Version           uint16                           `json:"version"`
	CompositionRecipe artifact.ID                      `json:"composition_recipe"`
	ExecutionRecipe   artifact.ID                      `json:"execution_recipe"`
	SourceModel       artifact.ID                      `json:"source_model"`
	TargetModel       artifact.ID                      `json:"target_model"`
	Task              recipe.Task                      `json:"task"`
	SourceContract    artifact.ID                      `json:"source_contract"`
	TargetContract    artifact.ID                      `json:"target_contract"`
	BridgeDefinitions []artifact.ID                    `json:"bridge_definitions"`
	BridgeWeights     []artifact.ID                    `json:"bridge_weights"`
	Operators         []bridgegraph.Operator           `json:"operators"`
	Capture           CompositionBoundary              `json:"capture"`
	Injection         CompositionBoundary              `json:"injection"`
	Sessions          modelrecipe.ComponentSessionPlan `json:"sessions"`
	TrainingPolicy    artifact.ID                      `json:"training_policy"`
	PromotionPolicy   artifact.ID                      `json:"promotion_policy"`
	Promotion         artifact.ID                      `json:"promotion"`
	CacheIdentity     artifact.ID                      `json:"cache_identity"`
	ID                artifact.ID                      `json:"-"`
}

type compositionCacheAuthority struct {
	SourceModel       artifact.ID            `json:"source_model"`
	SourceContract    artifact.ID            `json:"source_contract"`
	TargetContract    artifact.ID            `json:"target_contract"`
	BridgeDefinitions []artifact.ID          `json:"bridge_definitions"`
	BridgeWeights     []artifact.ID          `json:"bridge_weights"`
	Operators         []bridgegraph.Operator `json:"operators"`
}

var compositionExecutionPlanCodec = artifact.JSONDocumentCodec(
	"composition execution plan", artifact.KindProfile,
	CompositionExecutionPlanMediaType, CompositionExecutionPlanSchema,
	canonicalizeCompositionExecutionPlan,
	func(value CompositionExecutionPlan) artifact.ID { return value.ID },
	func(value *CompositionExecutionPlan, id artifact.ID) { value.ID = id },
	cloneCompositionExecutionPlan,
)

// CompileCompositionExecutionPlan resolves the scoped active alias and
// compiles its complete authority graph. No caller-supplied recipe or bridge
// can bypass RepoDB activation and promotion.
func CompileCompositionExecutionPlan(
	ctx context.Context,
	reader artifact.Reader,
	source, target artifact.ID,
	task recipe.Task,
) (CompositionExecutionPlan, error) {
	active, found, err := ActiveComposition(ctx, reader, source, target, task)
	if err != nil {
		return CompositionExecutionPlan{}, err
	}
	if !found {
		return CompositionExecutionPlan{}, errors.New("composition: active composition recipe is absent")
	}
	authority, err := loadCompositionAuthority(ctx, reader, active)
	if err != nil {
		return CompositionExecutionPlan{}, err
	}
	if err := validateCompositionAuthority(authority); err != nil {
		return CompositionExecutionPlan{}, err
	}
	sourceContent, err := authority.SourceContract.Content()
	if err != nil {
		return CompositionExecutionPlan{}, err
	}
	targetContent, err := authority.TargetContract.Content()
	if err != nil {
		return CompositionExecutionPlan{}, err
	}
	program, err := (bridgegraph.Compiler{}).Compile(authority.Bridge.Graph, sourceContent.Data, targetContent.Data)
	if err != nil {
		return CompositionExecutionPlan{}, err
	}
	capture, injection := program.Boundaries()
	sessions, err := modelrecipe.CompileDefinitionSessionPlan(ctx, reader, authority.Execution)
	if err != nil {
		return CompositionExecutionPlan{}, err
	}
	if err := validateCompositionSessionAuthority(
		sessions, authority.Execution.ID, source, target,
		active.SourceContract, active.TargetContract, active.BridgeWeights,
	); err != nil {
		return CompositionExecutionPlan{}, err
	}
	plan := CompositionExecutionPlan{
		Version:           CompositionExecutionPlanVersion,
		CompositionRecipe: active.ID, ExecutionRecipe: active.ExecutionRecipe,
		SourceModel: source, TargetModel: target, Task: task,
		SourceContract: active.SourceContract, TargetContract: active.TargetContract,
		BridgeDefinitions: []artifact.ID{active.BridgeDefinition},
		BridgeWeights:     []artifact.ID{active.BridgeWeights},
		Operators:         []bridgegraph.Operator{authority.Bridge.Graph.Operator},
		Capture:           compositionBoundary(capture), Injection: compositionBoundary(injection),
		Sessions:       sessions,
		TrainingPolicy: active.TrainingPolicy, PromotionPolicy: active.PromotionPolicy,
		Promotion: active.Promotion,
	}
	plan.CacheIdentity, err = compositionCacheIdentity(plan)
	if err != nil {
		return CompositionExecutionPlan{}, err
	}
	sealed, err := compositionExecutionPlanCodec.New(plan)
	if err != nil {
		return CompositionExecutionPlan{}, err
	}
	if authority.Bridge.Graph.Operator == bridgegraph.OperatorExternalAttention {
		if _, err := CompileExternalCrossAttentionPlan(sealed, authority.Bridge); err != nil {
			return CompositionExecutionPlan{}, err
		}
	}
	return sealed, nil
}

// Content returns the exact compiled composition plan document.
func (value CompositionExecutionPlan) Content() (artifact.Content, error) {
	return compositionExecutionPlanCodec.Content(value)
}

// Lineage binds a compiled plan to every authority selected through its
// active recipe, including policy and promotion identities.
func (value CompositionExecutionPlan) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.CompositionRecipe, value.ExecutionRecipe,
		value.SourceModel, value.TargetModel, value.SourceContract, value.TargetContract,
		value.TrainingPolicy, value.PromotionPolicy, value.Promotion,
	}
	parents = append(parents, value.BridgeDefinitions...)
	parents = append(parents, value.BridgeWeights...)
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

func canonicalizeCompositionExecutionPlan(value *CompositionExecutionPlan) error {
	if value == nil || value.Version != CompositionExecutionPlanVersion ||
		value.CompositionRecipe.Kind() != artifact.KindRecipe || value.ExecutionRecipe.Kind() != artifact.KindRecipe ||
		value.SourceModel.Kind() != artifact.KindModel || value.TargetModel.Kind() != artifact.KindModel ||
		value.SourceModel == value.TargetModel || !value.Task.Valid() ||
		value.SourceContract.Kind() != artifact.KindProfile || value.TargetContract.Kind() != artifact.KindProfile ||
		value.TrainingPolicy.Kind() != artifact.KindProfile || value.PromotionPolicy.Kind() != artifact.KindProfile ||
		value.Promotion.Kind() != artifact.KindEvidence || value.CacheIdentity.Kind() != artifact.KindProfile ||
		len(value.BridgeDefinitions) == 0 || len(value.BridgeDefinitions) != len(value.BridgeWeights) ||
		len(value.BridgeDefinitions) != len(value.Operators) {
		return errors.New("composition: invalid composition execution plan")
	}
	for index, definition := range value.BridgeDefinitions {
		if definition.Kind() != artifact.KindProfile || value.BridgeWeights[index].Kind() != artifact.KindAdapter ||
			!value.Operators[index].Valid() {
			return errors.New("composition: invalid ordered bridge program")
		}
	}
	if err := validateCompositionBoundary(value.Capture, value.SourceModel, value.SourceContract); err != nil {
		return err
	}
	if err := validateCompositionBoundary(value.Injection, value.TargetModel, value.TargetContract); err != nil {
		return err
	}
	if value.Sessions.Recipe != value.ExecutionRecipe || len(value.Sessions.Components) != tensor.PairedExtent ||
		value.Sessions.Components[tensor.FirstOffset].Model != value.SourceModel ||
		value.Sessions.Components[tensor.FirstOffset].Module != modelrecipe.ModuleCaptureRepresentation ||
		value.Sessions.Components[tensor.SingletonExtent].Model != value.TargetModel ||
		value.Sessions.Components[tensor.SingletonExtent].Module != modelrecipe.ModuleInjectRepresentation {
		return errors.New("composition: invalid ordered composition components")
	}
	if _, err := value.Sessions.Content(); err != nil {
		return err
	}
	if err := validateCompositionSessions(value.Sessions); err != nil {
		return err
	}
	identity, err := compositionCacheIdentity(*value)
	if err != nil || identity != value.CacheIdentity {
		return errors.Join(err, errors.New("composition: cache identity differs from immutable authorities"))
	}
	return nil
}

func compositionCacheIdentity(value CompositionExecutionPlan) (artifact.ID, error) {
	return artifact.JSONID(artifact.KindProfile, compositionCacheAuthority{
		SourceModel: value.SourceModel, SourceContract: value.SourceContract,
		TargetContract:    value.TargetContract,
		BridgeDefinitions: value.BridgeDefinitions, BridgeWeights: value.BridgeWeights,
		Operators: value.Operators,
	})
}

func validateCompositionBoundary(value CompositionBoundary, model, contract artifact.ID) error {
	if value.Model != model || value.Contract != contract || !checked.Nonzero(value.Channels) || value.Tap == "" {
		return errors.New("composition: compiled representation boundary differs")
	}
	return nil
}

func compositionBoundary(value bridgegraph.Boundary) CompositionBoundary {
	result := CompositionBoundary{
		Contract: value.Contract, Model: value.Model, Tap: value.Tap, Channels: value.Channels,
	}
	if value.Layer != nil {
		layer := *value.Layer
		result.Layer = &layer
	}
	return result
}

func validateCompositionSessions(value modelrecipe.ComponentSessionPlan) error {
	if len(value.Components) != tensor.PairedExtent {
		return errors.New("composition: residency components are incomplete")
	}
	source, target := value.Components[tensor.FirstOffset], value.Components[tensor.SingletonExtent]
	if source.Placement != recipe.PlacementDevice || target.Placement != recipe.PlacementDevice ||
		source.Session != recipe.SessionCapacity || target.Session != recipe.SessionRequest ||
		source.Residency != target.Residency || !compositionDeviceResidency(source.Residency) {
		return errors.New("composition: production representation path is not device-resident")
	}
	return nil
}

func validateCompositionSessionAuthority(
	value modelrecipe.ComponentSessionPlan,
	definition, source, target, sourceContract, targetContract, bridge artifact.ID,
) error {
	if err := validateCompositionSessions(value); err != nil {
		return err
	}
	sourceSession, targetSession := value.Components[tensor.FirstOffset], value.Components[tensor.SingletonExtent]
	expected, err := (modelrecipe.RepresentationBridgeCompiler{
		SourcePlacement: sourceSession.Placement, TargetPlacement: targetSession.Placement,
		SourceResidency: sourceSession.Residency, TargetResidency: targetSession.Residency,
		SourceSession: sourceSession.Session, TargetSession: targetSession.Session,
	}).Definition(source, target, sourceContract, targetContract, bridge)
	if err != nil || expected.ID != definition {
		return errors.Join(err, errors.New("composition: component session authority differs from execution recipe"))
	}
	return nil
}

func compositionDeviceResidency(value recipe.ResidencyPolicy) bool {
	return value == recipe.ResidencyDeviceF32 || value == recipe.ResidencyDeviceNative ||
		value == recipe.ResidencyDeviceNativeBF16
}

func cloneCompositionExecutionPlan(value CompositionExecutionPlan) CompositionExecutionPlan {
	value.BridgeDefinitions = append([]artifact.ID(nil), value.BridgeDefinitions...)
	value.BridgeWeights = append([]artifact.ID(nil), value.BridgeWeights...)
	value.Operators = append([]bridgegraph.Operator(nil), value.Operators...)
	value.Sessions.Components = append([]modelrecipe.ComponentSession(nil), value.Sessions.Components...)
	if value.Capture.Layer != nil {
		layer := *value.Capture.Layer
		value.Capture.Layer = &layer
	}
	if value.Injection.Layer != nil {
		layer := *value.Injection.Layer
		value.Injection.Layer = &layer
	}
	return value
}
