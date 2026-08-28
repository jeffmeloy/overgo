package workflowcontract

import (
	"cmp"
	"context"
	"errors"
	"slices"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const (
	// AutomationExecutionPlanVersion is the compiled plan revision.
	AutomationExecutionPlanVersion = artifact.InitialDocumentVersion
	// AutomationExecutionPlanMediaType identifies compiled plan documents.
	AutomationExecutionPlanMediaType = "application/vnd.overgo.automation-execution-plan+json"
	// AutomationExecutionPlanSchema identifies the compiled plan contract.
	AutomationExecutionPlanSchema = "overgo/automation-execution-plan/v1"
)

// AutomationExecutionPlan is the complete immutable authority accepted by
// automation runtimes. It contains identities selected through the active
// alias, never caller-supplied substitutes.
type AutomationExecutionPlan struct {
	Version           uint16                           `json:"version"`
	Name              string                           `json:"name"`
	Definition        artifact.ID                      `json:"definition"`
	Activation        artifact.ID                      `json:"activation"`
	Recipe            artifact.ID                      `json:"recipe"`
	Task              recipe.Task                      `json:"task"`
	OperationPolicy   artifact.ID                      `json:"operation_policy"`
	Trigger           recipe.AutomationTriggerPolicy   `json:"trigger"`
	Delivery          recipe.AutomationDeliveryPolicy  `json:"delivery"`
	Datasets          []artifact.ID                    `json:"datasets,omitempty"`
	CapabilityBundles []artifact.ID                    `json:"capability_bundles,omitempty"`
	Resources         modelrecipe.ComponentSessionPlan `json:"resources"`
	CacheIdentity     artifact.ID                      `json:"cache_identity"`
	ID                artifact.ID                      `json:"-"`
}

type automationCacheAuthority struct {
	Definition        artifact.ID   `json:"definition"`
	Activation        artifact.ID   `json:"activation"`
	Recipe            artifact.ID   `json:"recipe"`
	OperationPolicy   artifact.ID   `json:"operation_policy"`
	Trigger           artifact.ID   `json:"trigger"`
	Delivery          artifact.ID   `json:"delivery"`
	Datasets          []artifact.ID `json:"datasets,omitempty"`
	CapabilityBundles []artifact.ID `json:"capability_bundles,omitempty"`
	Resources         artifact.ID   `json:"resources"`
}

var automationExecutionPlanCodec = artifact.JSONDocumentCodec(
	"automation execution plan", artifact.KindProfile,
	AutomationExecutionPlanMediaType, AutomationExecutionPlanSchema,
	canonicalizeAutomationExecutionPlan,
	func(value AutomationExecutionPlan) artifact.ID { return value.ID },
	func(value *AutomationExecutionPlan, id artifact.ID) { value.ID = id },
	func(value AutomationExecutionPlan) AutomationExecutionPlan {
		value.Datasets = slices.Clone(value.Datasets)
		value.CapabilityBundles = slices.Clone(value.CapabilityBundles)
		value.Delivery.Destinations = slices.Clone(value.Delivery.Destinations)
		value.Resources.Components = slices.Clone(value.Resources.Components)
		return value
	},
)

// AutomationCompiler resolves active authority and compiles it with the one
// recipe catalog supplied by the owning runtime.
type AutomationCompiler struct {
	Repository artifact.Repository
	Catalog    *recipe.Catalog
}

// Compile resolves and seals one active automation execution plan.
func (compiler AutomationCompiler) Compile(ctx context.Context, name string) (AutomationExecutionPlan, error) {
	if ctx == nil || compiler.Repository == nil || compiler.Catalog == nil {
		return AutomationExecutionPlan{}, errors.New("workflow contract: automation compiler authority is absent")
	}
	active, found, err := (runrecord.AutomationAuthority{Repository: compiler.Repository}).Resolve(ctx, name)
	if err != nil || !found {
		return AutomationExecutionPlan{}, errors.Join(errors.New("workflow contract: active automation is absent"), err)
	}
	definition, err := recipe.RequireDefinition(ctx, compiler.Repository, active.Definition.Recipe)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	program, err := recipe.CompileProgram(definition, compiler.Catalog)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	if err := requireAutomationDependencies(definition, active.Definition); err != nil {
		return AutomationExecutionPlan{}, err
	}
	trigger, err := recipe.RequireAutomationTriggerPolicy(ctx, compiler.Repository, active.Definition.TriggerPolicy)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	if err := requireWebhookWorkflow(trigger, definition.ID); err != nil {
		return AutomationExecutionPlan{}, err
	}
	delivery, err := recipe.RequireAutomationDeliveryPolicy(ctx, compiler.Repository, active.Definition.DeliveryPolicy)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	dependencies := append(slices.Clone(active.Definition.Datasets), active.Definition.CapabilityBundles...)
	if delivery.Tool.Valid() {
		dependencies = append(dependencies, delivery.Tool, delivery.Authorization)
	}
	if err := requireAutomationArtifacts(ctx, compiler.Repository, dependencies); err != nil {
		return AutomationExecutionPlan{}, err
	}
	if err := requireAutomationDeliveryManual(ctx, compiler.Repository, delivery); err != nil {
		return AutomationExecutionPlan{}, err
	}
	policy, err := modelrecipe.ResolveRuntimePolicy(ctx, compiler.Repository, definition)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	resources, err := modelrecipe.CompileComponentSessionPlan(ctx, compiler.Repository, program)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	plan := AutomationExecutionPlan{
		Version: AutomationExecutionPlanVersion, Name: active.Definition.Name,
		Definition: active.Definition.ID, Activation: active.Activation.ID,
		Recipe: definition.ID, Task: definition.Task, OperationPolicy: policy.ID,
		Trigger: trigger, Delivery: delivery,
		Datasets:          slices.Clone(active.Definition.Datasets),
		CapabilityBundles: slices.Clone(active.Definition.CapabilityBundles), Resources: resources,
	}
	plan.CacheIdentity, err = automationCacheIdentity(plan)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	return automationExecutionPlanCodec.New(plan)
}

// Require loads one previously compiled plan for restart recovery and verifies
// that every exact authority it names remains available and compatible.
func (compiler AutomationCompiler) Require(ctx context.Context, id artifact.ID) (AutomationExecutionPlan, error) {
	if ctx == nil || compiler.Repository == nil || compiler.Catalog == nil {
		return AutomationExecutionPlan{}, errors.New("workflow contract: automation compiler authority is absent")
	}
	plan, err := automationExecutionPlanCodec.Require(ctx, compiler.Repository, id)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	definition, err := recipe.RequireDefinition(ctx, compiler.Repository, plan.Recipe)
	if err != nil || definition.Task != plan.Task {
		return AutomationExecutionPlan{}, errors.Join(errors.New("workflow contract: automation recovery recipe differs"), err)
	}
	program, err := recipe.CompileProgram(definition, compiler.Catalog)
	if err != nil {
		return AutomationExecutionPlan{}, err
	}
	resources, err := modelrecipe.CompileComponentSessionPlan(ctx, compiler.Repository, program)
	if err != nil || resources.Identity != plan.Resources.Identity {
		return AutomationExecutionPlan{}, errors.Join(errors.New("workflow contract: automation recovery resources differ"), err)
	}
	policy, err := modelrecipe.ResolveRuntimePolicy(ctx, compiler.Repository, definition)
	if err != nil || policy.ID != plan.OperationPolicy {
		return AutomationExecutionPlan{}, errors.Join(errors.New("workflow contract: automation recovery policy differs"), err)
	}
	trigger, err := recipe.RequireAutomationTriggerPolicy(ctx, compiler.Repository, plan.Trigger.ID)
	if err != nil || trigger.ID != plan.Trigger.ID {
		return AutomationExecutionPlan{}, errors.Join(errors.New("workflow contract: automation recovery trigger differs"), err)
	}
	delivery, err := recipe.RequireAutomationDeliveryPolicy(ctx, compiler.Repository, plan.Delivery.ID)
	if err != nil || delivery.ID != plan.Delivery.ID {
		return AutomationExecutionPlan{}, errors.Join(errors.New("workflow contract: automation recovery delivery differs"), err)
	}
	if err := requireAutomationDeliveryManual(ctx, compiler.Repository, delivery); err != nil {
		return AutomationExecutionPlan{}, err
	}
	return plan, nil
}

// Content returns exact repository content for the compiled plan.
func (value AutomationExecutionPlan) Content() (artifact.Content, error) {
	return automationExecutionPlanCodec.Content(value)
}

// ValidateIdentity verifies all compiled-plan invariants and identity.
func (value AutomationExecutionPlan) ValidateIdentity() error {
	return automationExecutionPlanCodec.ValidateIdentity(value)
}

// Lineage binds the plan to every immutable selected authority.
func (value AutomationExecutionPlan) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Definition, value.Activation, value.Recipe, value.OperationPolicy,
		value.Trigger.ID, value.Delivery.ID, value.Resources.Identity,
	}
	parents = append(parents, value.Datasets...)
	parents = append(parents, value.CapabilityBundles...)
	return artifact.DependencyLineage(value.ID, parents...)
}

// ExecutionID derives the stable operation identity for one plan and caller
// idempotency key. Policy changes therefore cannot resume an older operation.
func (value AutomationExecutionPlan) ExecutionID(key string) (artifact.ID, error) {
	if automationExecutionPlanCodec.ValidateIdentity(value) != nil || key == "" {
		return artifact.ID{}, errors.New("workflow contract: invalid automation execution identity input")
	}
	return artifact.JSONID(artifact.KindEvidence, struct {
		Plan artifact.ID `json:"plan"`
		Key  string      `json:"key"`
	}{Plan: value.ID, Key: key})
}

func canonicalizeAutomationExecutionPlan(value *AutomationExecutionPlan) error {
	if value == nil || value.Trigger.Version != recipe.AutomationTriggerPolicyVersion ||
		value.Delivery.Version != recipe.AutomationDeliveryPolicyVersion {
		return errors.New("workflow contract: invalid automation execution plan")
	}
	trigger, triggerErr := value.Trigger.Identify()
	delivery, deliveryErr := value.Delivery.Identify()
	if triggerErr != nil || deliveryErr != nil {
		return errors.Join(errors.New("workflow contract: invalid automation policy"), triggerErr, deliveryErr)
	}
	value.Trigger, value.Delivery = trigger, delivery
	if value == nil || value.Version != AutomationExecutionPlanVersion || value.Name == "" ||
		value.Definition.Kind() != artifact.KindRecipe || value.Activation.Kind() != artifact.KindEvidence ||
		value.Recipe.Kind() != artifact.KindRecipe || !value.Task.Valid() ||
		value.OperationPolicy.Kind() != artifact.KindProfile || value.CacheIdentity.Kind() != artifact.KindProfile ||
		value.Trigger.ID.Kind() != artifact.KindProfile || value.Delivery.ID.Kind() != artifact.KindProfile ||
		value.Resources.Identity.Kind() != artifact.KindProfile || value.Resources.Recipe != value.Recipe {
		return errors.New("workflow contract: invalid automation execution plan")
	}
	if value.Trigger.ValidateIdentity() != nil || value.Delivery.ValidateIdentity() != nil {
		return errors.New("workflow contract: automation policy identity differs")
	}
	if err := requireWebhookWorkflow(value.Trigger, value.Recipe); err != nil {
		return err
	}
	if _, err := value.Resources.Content(); err != nil {
		return err
	}
	want, err := automationCacheIdentity(*value)
	if err != nil || want != value.CacheIdentity {
		return errors.Join(errors.New("workflow contract: automation cache identity differs"), err)
	}
	return nil
}

func requireWebhookWorkflow(trigger recipe.AutomationTriggerPolicy, workflow artifact.ID) error {
	if trigger.Kind == recipe.AutomationTriggerWebhook &&
		(trigger.Webhook == nil || trigger.Webhook.Workflow != workflow) {
		return errors.New("workflow contract: webhook workflow differs from automation recipe")
	}
	return nil
}

func automationCacheIdentity(value AutomationExecutionPlan) (artifact.ID, error) {
	return artifact.JSONID(artifact.KindProfile, automationCacheAuthority{
		Definition: value.Definition, Activation: value.Activation, Recipe: value.Recipe,
		OperationPolicy: value.OperationPolicy, Trigger: value.Trigger.ID, Delivery: value.Delivery.ID,
		Datasets: value.Datasets, CapabilityBundles: value.CapabilityBundles, Resources: value.Resources.Identity,
	})
}

func requireAutomationDependencies(definition recipe.Definition, automation recipe.AutomationDefinition) error {
	datasets := make([]artifact.ID, 0)
	bundles := make([]artifact.ID, 0)
	for _, dependency := range definition.Dependencies {
		switch dependency.Role {
		case recipe.DependencyDataset:
			datasets = append(datasets, dependency.Artifact)
		case recipe.DependencyCapabilityBundle:
			bundles = append(bundles, dependency.Artifact)
		}
	}
	slices.SortFunc(datasets, func(a, b artifact.ID) int { return cmp.Compare(a.String(), b.String()) })
	slices.SortFunc(bundles, func(a, b artifact.ID) int { return cmp.Compare(a.String(), b.String()) })
	if !slices.Equal(datasets, automation.Datasets) || !slices.Equal(bundles, automation.CapabilityBundles) {
		return errors.New("workflow contract: automation dependencies differ from recipe")
	}
	return nil
}

func requireAutomationArtifacts(ctx context.Context, reader artifact.Reader, ids []artifact.ID) error {
	for _, id := range ids {
		descriptor, found, err := reader.Artifact(ctx, id)
		if err != nil || !found || descriptor.ID != id || descriptor.Size == 0 {
			return errors.Join(errors.New("workflow contract: automation dependency artifact is absent"), err)
		}
	}
	return nil
}

func requireAutomationDeliveryManual(
	ctx context.Context,
	reader artifact.Reader,
	delivery recipe.AutomationDeliveryPolicy,
) error {
	if delivery.Kind != recipe.AutomationDeliveryTool {
		return nil
	}
	manual, err := agenttool.RequireManual(ctx, reader, delivery.Tool)
	if err != nil || manual.Effect != agenttool.EffectMutation {
		return errors.Join(errors.New("workflow contract: delivery requires an effect-classified mutation tool"), err)
	}
	registered, found, err := artifact.ResolveAlias(ctx, reader, agenttool.RegisteredAlias(manual.Name))
	if err != nil || !found || registered != manual.ID {
		return errors.Join(errors.New("workflow contract: delivery tool is not registered exactly"), err)
	}
	fields := make(map[string]agenttool.Field, len(manual.Arguments))
	for _, field := range manual.Arguments {
		fields[field.Name] = field
	}
	destination := fields[delivery.DestinationArgument]
	payload := fields[delivery.PayloadArgument]
	idempotency := fields[delivery.IdempotencyArgument]
	if !destination.Required || destination.Kind != agenttool.FieldString ||
		!payload.Required || payload.Kind != agenttool.FieldObject ||
		!idempotency.Required || idempotency.Kind != agenttool.FieldString {
		return errors.New("workflow contract: delivery tool arguments differ from policy")
	}
	return nil
}
