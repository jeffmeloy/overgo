package server

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"time"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowcontract"
	"overgo/internal/workflowruntime"
)

// AutomationInventoryEntry projects active immutable authority and any exact
// compile refusal; unavailable plans are visible but never executable.
type AutomationInventoryEntry struct {
	Name       string                                    `json:"name"`
	Definition artifact.ID                               `json:"definition"`
	Activation artifact.ID                               `json:"activation"`
	Plan       *workflowcontract.AutomationExecutionPlan `json:"plan,omitempty"`
	Refusal    string                                    `json:"refusal,omitempty"`
}

// AutomationDefinitionInput is the typed editing boundary shared by API and GUI.
type AutomationDefinitionInput struct {
	Name              string                          `json:"name"`
	Recipe            artifact.ID                     `json:"recipe"`
	Trigger           recipe.AutomationTriggerPolicy  `json:"trigger"`
	Delivery          recipe.AutomationDeliveryPolicy `json:"delivery"`
	Datasets          []artifact.ID                   `json:"datasets,omitempty"`
	CapabilityBundles []artifact.ID                   `json:"capability_bundles,omitempty"`
}

// AutomationExecutionInput is one manual or scheduled invocation.
type AutomationExecutionInput struct {
	Name        string                     `json:"name"`
	Key         string                     `json:"key,omitempty"`
	Destination string                     `json:"destination,omitempty"`
	Inputs      map[string]json.RawMessage `json:"inputs"`
}

// AutomationHistoryEntry is the durable run projection used by API and GUI.
type AutomationHistoryEntry struct {
	ID      artifact.ID       `json:"id"`
	Recipe  artifact.ID       `json:"recipe"`
	Outcome runrecord.Outcome `json:"outcome"`
	Failure string            `json:"failure,omitempty"`
	Inputs  []artifact.ID     `json:"inputs,omitempty"`
	Outputs []artifact.ID     `json:"outputs,omitempty"`
}

// AutomationWorkspaceAPI is the server protocol boundary. Implementations use
// recipe/runrecord/workflowruntime owners; handlers only decode and project.
type AutomationWorkspaceAPI interface {
	AutomationInventory(context.Context) ([]AutomationInventoryEntry, error)
	PublishAutomationDefinition(context.Context, AutomationDefinitionInput) (recipe.AutomationDefinition, error)
	ActivateAutomation(context.Context, artifact.ID) (runrecord.ActiveAutomation, error)
	RunAutomation(context.Context, *operation.Manager, AutomationExecutionInput) (workflowruntime.AutomationExecution, error)
	ScheduleAutomation(context.Context, *operation.Manager, AutomationExecutionInput) (workflowruntime.AutomationExecution, bool, error)
	AutomationHistory(context.Context) ([]AutomationHistoryEntry, error)
}

// AutomationWorkspace composes existing authorities for server and GUI use.
type AutomationWorkspace struct {
	store    *overgodb.Store
	catalog  *recipe.Catalog
	adapters map[recipe.ModuleID]workflowruntime.Adapter
	tools    *agenttool.Executor
	limit    int
	clock    workflowruntime.AutomationClock
}

type systemAutomationClock struct{}

// Now returns the current UTC scheduler time.
func (systemAutomationClock) Now() time.Time { return time.Now().UTC() }

// AutomationWorkspaceConfig binds protocol projection to shared runtime owners.
type AutomationWorkspaceConfig struct {
	Store    *overgodb.Store
	Catalog  *recipe.Catalog
	Adapters map[recipe.ModuleID]workflowruntime.Adapter
	Tools    *agenttool.Executor
	Limit    int
}

// Open validates and creates one automation workspace.
func (config AutomationWorkspaceConfig) Open() (*AutomationWorkspace, error) {
	if config.Store == nil || config.Catalog == nil || config.Limit <= 0 {
		return nil, errors.New("automation workspace: repository, catalog, and positive history limit required")
	}
	return &AutomationWorkspace{
		store: config.Store, catalog: config.Catalog, adapters: cloneAutomationAdapters(config.Adapters),
		tools: config.Tools, limit: config.Limit, clock: systemAutomationClock{},
	}, nil
}

// AutomationInventory returns active aliases and compile/refusal truth.
func (workspace *AutomationWorkspace) AutomationInventory(ctx context.Context) ([]AutomationInventoryEntry, error) {
	if workspace == nil || ctx == nil {
		return nil, errors.New("automation workspace: inventory authority is absent")
	}
	result, err := workspace.store.Query(ctx, overgodb.Query{
		MaxResults: workspace.limit, Projection: overgodb.ProjectAliases,
	})
	if err != nil {
		return nil, err
	}
	entries := make([]AutomationInventoryEntry, 0)
	authority := runrecord.AutomationAuthority{Repository: workspace.store}
	compiler := workflowcontract.AutomationCompiler{Repository: workspace.store, Catalog: workspace.catalog}
	for _, alias := range result.Aliases {
		name, found := strings.CutPrefix(alias.Name, runrecord.AutomationActiveAliasRoot)
		if !found {
			continue
		}
		active, resolved, resolveErr := authority.Resolve(ctx, name)
		if resolveErr != nil || !resolved {
			return nil, errors.Join(errors.New("automation workspace: active alias differs"), resolveErr)
		}
		entry := AutomationInventoryEntry{
			Name: name, Definition: active.Definition.ID, Activation: active.Activation.ID,
		}
		plan, compileErr := compiler.Compile(ctx, name)
		if compileErr != nil {
			entry.Refusal = compileErr.Error()
		} else {
			entry.Plan = &plan
		}
		entries = append(entries, entry)
	}
	slices.SortFunc(entries, func(left, right AutomationInventoryEntry) int {
		return strings.Compare(left.Name, right.Name)
	})
	return entries, nil
}

// PublishAutomationDefinition commits typed policies and immutable definition.
func (workspace *AutomationWorkspace) PublishAutomationDefinition(
	ctx context.Context,
	input AutomationDefinitionInput,
) (recipe.AutomationDefinition, error) {
	trigger, err := input.Trigger.Identify()
	if err != nil {
		return recipe.AutomationDefinition{}, err
	}
	if trigger.Kind == recipe.AutomationTriggerWebhook && trigger.Webhook.Workflow != input.Recipe {
		return recipe.AutomationDefinition{}, errors.New("server: webhook trigger names a different workflow")
	}
	delivery, err := input.Delivery.Identify()
	if err != nil {
		return recipe.AutomationDefinition{}, err
	}
	definition, err := recipe.NewAutomationDefinition(recipe.AutomationDefinition{
		Name: input.Name, Recipe: input.Recipe, TriggerPolicy: trigger.ID, DeliveryPolicy: delivery.ID,
		Datasets: input.Datasets, CapabilityBundles: input.CapabilityBundles,
	})
	if err != nil {
		return recipe.AutomationDefinition{}, err
	}
	triggerContent, err := trigger.ArtifactContent()
	if err != nil {
		return recipe.AutomationDefinition{}, err
	}
	deliveryContent, err := delivery.ArtifactContent()
	if err != nil {
		return recipe.AutomationDefinition{}, err
	}
	definitionContent, err := definition.ArtifactContent()
	if err != nil {
		return recipe.AutomationDefinition{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"automation/workspace/definition/"+definition.ID.String(),
		[]artifact.Content{triggerContent, deliveryContent, definitionContent}, append(definition.Lineage(), trigger.Lineage()...), nil,
	)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, workspace.store, batch)
	}
	if errors.Is(err, artifact.ErrNoChange) {
		err = nil
	}
	return definition, err
}

// ActivateAutomation advances the named alias with request evidence.
func (workspace *AutomationWorkspace) ActivateAutomation(
	ctx context.Context,
	definitionID artifact.ID,
) (runrecord.ActiveAutomation, error) {
	definition, err := recipe.RequireAutomationDefinition(ctx, workspace.store, definitionID)
	if err != nil {
		return runrecord.ActiveAutomation{}, err
	}
	authorityContent, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindEvidence, "overgo.automation-workspace-activation.v1"),
		struct {
			Definition artifact.ID `json:"definition"`
		}{Definition: definition.ID},
	)
	if err != nil {
		return runrecord.ActiveAutomation{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"automation/workspace/activation-authority/"+authorityContent.Descriptor.ID.String(),
		[]artifact.Content{authorityContent}, artifact.DependencyLineage(authorityContent.Descriptor.ID, definition.ID), nil,
	)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, workspace.store, batch)
	}
	if err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return runrecord.ActiveAutomation{}, err
	}
	return (runrecord.AutomationAuthority{Repository: workspace.store}).Activate(
		ctx, "automation/workspace/activate/"+definition.ID.String(), definition, authorityContent.Descriptor.ID,
	)
}

// RunAutomation admits manual execution to the caller's common operation manager.
func (workspace *AutomationWorkspace) RunAutomation(
	ctx context.Context,
	operations *operation.Manager,
	input AutomationExecutionInput,
) (workflowruntime.AutomationExecution, error) {
	values, err := workspace.automationInputs(ctx, input.Name, input.Inputs)
	if err != nil {
		return workflowruntime.AutomationExecution{}, err
	}
	runtime := workspace.runtime(operations)
	return runtime.SubmitManualTo(ctx, input.Name, input.Key, input.Destination, values)
}

// ScheduleAutomation scans and admits one due schedule claim.
func (workspace *AutomationWorkspace) ScheduleAutomation(
	ctx context.Context,
	operations *operation.Manager,
	input AutomationExecutionInput,
) (workflowruntime.AutomationExecution, bool, error) {
	values, err := workspace.automationInputs(ctx, input.Name, input.Inputs)
	if err != nil {
		return workflowruntime.AutomationExecution{}, false, err
	}
	return workspace.runtime(operations).ScheduleAutomation(ctx, input.Name, workspace.clock, values)
}

// AutomationHistory returns durable runs belonging to active automation recipes.
func (workspace *AutomationWorkspace) AutomationHistory(ctx context.Context) ([]AutomationHistoryEntry, error) {
	inventory, err := workspace.AutomationInventory(ctx)
	if err != nil {
		return nil, err
	}
	recipes := make(map[artifact.ID]struct{}, len(inventory))
	for _, entry := range inventory {
		if entry.Plan != nil {
			recipes[entry.Plan.Recipe] = struct{}{}
		}
	}
	runs := make([]AutomationHistoryEntry, 0, workspace.limit)
	_, err = overgodb.VisitDecodedDocuments(ctx, workspace.store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{
			{Kind: artifact.KindRun, MediaType: runrecord.RunMediaType, Schema: runrecord.LegacyRunSchema},
			{Kind: artifact.KindRun, MediaType: runrecord.RunMediaType, Schema: runrecord.RunSchema},
		},
		Order: overgodb.DocumentNewestFirst, MaxResults: workspace.limit,
	}, runrecord.ParseRun, func(_ overgodb.DocumentView, run runrecord.Run) error {
		if _, found := recipes[run.Recipe]; found {
			runs = append(runs, AutomationHistoryEntry{
				ID: run.ID, Recipe: run.Recipe, Outcome: run.Outcome, Failure: run.Failure,
				Inputs: slices.Clone(run.Inputs), Outputs: slices.Clone(run.Outputs),
			})
		}
		return nil
	})
	return runs, err
}

func (workspace *AutomationWorkspace) automationInputs(
	ctx context.Context,
	name string,
	raw map[string]json.RawMessage,
) (map[recipe.PortName]workflowruntime.Value, error) {
	active, found, err := (runrecord.AutomationAuthority{Repository: workspace.store}).Resolve(ctx, name)
	if err != nil || !found {
		return nil, errors.Join(errors.New("automation workspace: active definition is absent"), err)
	}
	definition, err := recipe.RequireDefinition(ctx, workspace.store, active.Definition.Recipe)
	if err != nil {
		return nil, err
	}
	if len(raw) != len(definition.Inputs) {
		return nil, errors.New("automation workspace: input set differs")
	}
	values := make(map[recipe.PortName]workflowruntime.Value, len(raw))
	for _, input := range definition.Inputs {
		encoded, found := raw[string(input.Name)]
		if !found {
			return nil, errors.New("automation workspace: required input is absent")
		}
		var value any
		if err := json.Unmarshal(encoded, &value); err != nil {
			return nil, err
		}
		values[input.Name] = workflowruntime.Value{
			Kind: input.Data, Items: []workflowruntime.Datum{{Value: value}},
		}
	}
	return values, nil
}

func (workspace *AutomationWorkspace) runtime(operations *operation.Manager) workflowruntime.AutomationRuntime {
	return workflowruntime.AutomationRuntime{
		Store: workspace.store, Operations: operations, Catalog: workspace.catalog,
		Adapters: cloneAutomationAdapters(workspace.adapters), Tools: workspace.tools,
	}
}

func cloneAutomationAdapters(values map[recipe.ModuleID]workflowruntime.Adapter) map[recipe.ModuleID]workflowruntime.Adapter {
	cloned := make(map[recipe.ModuleID]workflowruntime.Adapter, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}
