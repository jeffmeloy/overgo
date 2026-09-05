package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/discovery"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// StoreGenerationWorkspace is the generation workspace over the store
// (professional GUI campaign, gui-generation-workspace): every active
// media recipe the store declares lists as a capability with the controls
// its request declares and the media inputs its recipe binds, executes
// through the same executors the recipe command uses, and publishes its
// output as a media artifact. Any launcher gets it with the store; no
// flag and no model name is involved.
type StoreGenerationWorkspace struct {
	store   *overgodb.Store
	catalog map[recipe.Task]mediacapability.Capability
	limit   int
}

// NewStoreGenerationWorkspace binds the workspace to the store and to the
// capability catalog the recipe command shares.
func NewStoreGenerationWorkspace(store *overgodb.Store, catalog map[recipe.Task]mediacapability.Capability, limit int) *StoreGenerationWorkspace {
	return &StoreGenerationWorkspace{store: store, catalog: catalog, limit: limit}
}

// generationTasks are the media tasks a generation capability serves.
var generationTasks = []recipe.Task{recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskVideoEdit, recipe.TaskSpeech}

// generationRunKeyPrefix roots every generation run record's batch key.
const generationRunKeyPrefix = "generation/run/"

// WorkflowCapabilities lists one capability per active media recipe with
// present bytes, controls declared by the executor's request type, and a
// refusal on the entry when its request carries what a page cannot type.
func (workspace *StoreGenerationWorkspace) WorkflowCapabilities(ctx context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || kind != WorkflowGeneration {
		return nil, nil
	}
	memo := discovery.LoadMemo(ctx, workspace.store)
	entries, _, err := discovery.CapabilityCatalog(ctx, workspace.store, workspace.limit, memo)
	if err != nil {
		return nil, err
	}
	var capabilities []WorkflowCapability
	for _, entry := range entries {
		if !entry.Present {
			continue
		}
		for _, capability := range entry.Capabilities {
			if capability.Stale != "" || !slices.Contains(generationTasks, capability.Task) {
				continue
			}
			if _, known := workspace.catalog[capability.Task]; !known {
				continue
			}
			selection, err := modelrecipe.ResolveActiveExecution(ctx, workspace.store, entry.Model, capability.Task, modelrecipe.SessionWarm)
			if err != nil {
				return nil, fmt.Errorf("generation workspace: %s for %s: %w", capability.Task, entry.Model, err)
			}
			program := selection.Program
			definition := program.Definition()
			stages := program.Stages()
			if len(stages) == 0 {
				continue
			}
			directory, _ := workspace.executionPath(ctx, entry.Model, entry.Location)
			controls, refusal := mediacapability.Controls(stages[0].Module.ID, directory)
			declared := make([]WorkflowControl, 0, len(controls))
			for _, control := range controls {
				declared = append(declared, WorkflowControl{Name: control.Name, Type: WorkflowControlType(control.Type), Required: control.Required, Choices: control.Choices})
			}
			capabilities = append(capabilities, WorkflowCapability{
				Task: capability.Task, Recipe: definition.ID, Stages: stages,
				Inputs: definition.Inputs, Outputs: definition.Outputs, Controls: declared,
				Model: entry.Model, Name: filepath.Base(entry.Location), Location: entry.Location, Refusal: refusal,
			})
		}
	}
	return capabilities, nil
}

// ExecuteWorkflow runs the named recipe against its model bytes with the
// request the page typed, publishes the output as a media artifact and a
// run record, and returns both as the operation's receipt.
func (workspace *StoreGenerationWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != WorkflowGeneration {
		return operation.Completion{}, errors.New("generation workspace: workflow is not admitted")
	}
	capabilities, err := workspace.WorkflowCapabilities(ctx, kind)
	if err != nil {
		return operation.Completion{}, err
	}
	var selected *WorkflowCapability
	for index := range capabilities {
		if capabilities[index].Task == task && capabilities[index].Recipe == recipeID {
			selected = &capabilities[index]
			break
		}
	}
	if selected == nil {
		return operation.Completion{}, fmt.Errorf("generation workspace: recipe %s is not an active %s capability", recipeID, task)
	}
	if selected.Refusal != "" {
		return operation.Completion{}, errors.New("generation workspace: " + selected.Refusal)
	}
	path, err := workspace.executionPath(ctx, selected.Model, selected.Location)
	if err != nil {
		return operation.Completion{}, err
	}
	selection, err := modelrecipe.ResolveActiveExecution(ctx, workspace.store, selected.Model, task, modelrecipe.SessionWarm)
	if err != nil {
		return operation.Completion{}, err
	}
	output, err := workspace.catalog[task].Execute(ctx, workspace.store, path, selection, string(raw))
	if err != nil {
		return workspace.fail(ctx, recipeID, err)
	}
	content, err := mediacapability.OutputContent(capabilityruntime.Unwrap(output))
	if err != nil {
		return workspace.fail(ctx, recipeID, err)
	}
	run, err := runrecord.NewRun(recipeID, runrecord.OutcomeSucceeded, nil, []artifact.ID{content.Descriptor.ID}, "")
	if err != nil {
		return operation.Completion{}, err
	}
	batch, err := run.Batch(generationRunKeyPrefix + run.ID.String())
	if err != nil {
		return operation.Completion{}, err
	}
	batch.Contents = append(batch.Contents, content)
	if _, err := artifact.CommitBatch(ctx, workspace.store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return operation.Completion{}, err
	}
	reporter.Publishing()
	return operation.Completion{Run: run.ID, Outputs: []artifact.ID{content.Descriptor.ID}}, nil
}

// executionPath names the model directory an executor reads: every media
// capability resolves a repository directory (config beside weights), so
// the recorded directory location serves when the store holds one, else
// the directory holding the present file the catalog resolved.
func (workspace *StoreGenerationWorkspace) executionPath(ctx context.Context, model artifact.ID, location string) (string, error) {
	if path, err := artifact.AvailablePath(ctx, workspace.store, model, artifact.LocationDirectory); err == nil {
		return path, nil
	}
	if location == "" {
		return "", fmt.Errorf("generation workspace: model %s has no bytes on disk", model)
	}
	return filepath.Dir(location), nil
}

// fail records the failed generation as a run so the operation carries a
// durable receipt of what did not happen.
func (workspace *StoreGenerationWorkspace) fail(ctx context.Context, recipeID artifact.ID, cause error) (operation.Completion, error) {
	return failWorkflow(ctx, workspace.store, recipeID, nil, "generation_failed", cause)
}
