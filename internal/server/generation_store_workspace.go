package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingattempt"
)

// generationCatalog is what the store generation workspace needs from the
// media capability catalog, supplied by the assembly root so this package
// names no media executor package: one executor per media task, the
// controls a stage's request declares for a model directory, and the
// media artifact an executor's output becomes.
type generationCatalog struct {
	Execute       map[recipe.Task]capabilityruntime.Executor
	Controls      func(entry recipe.ModuleID, directory string) ([]workflowControl, string)
	OutputContent func(output any) (artifact.Content, error)
}

// BindGenerationCatalog builds the catalog from a capability catalog's
// executors, its control declarations and its output publisher, so the
// launcher and the tests bind the same shape once while the catalog's own
// types stay outside this package: a capability names its executor and a
// control reports its declaration as one tuple.
func BindGenerationCatalog[
	Capability interface {
		Executor() capabilityruntime.Executor
	},
	Control interface {
		Declared() (name, kind string, required bool, choices []string)
		// Slot reports an artifact control's label and media kind (empty for a typed field).
		Slot() (label, media string)
		// Bounded reports a numeric control's declared default, step and rate; declared is false without them.
		Bounded() (defaultValue, step, rate int, declared bool)
	},
](
	catalog map[recipe.Task]Capability,
	controls func(recipe.ModuleID, string) ([]Control, string),
	output func(any) (artifact.Content, error),
) generationCatalog {
	execute := make(map[recipe.Task]capabilityruntime.Executor, len(catalog))
	for task, capability := range catalog {
		execute[task] = capability.Executor()
	}
	return generationCatalog{
		Execute: execute,
		Controls: func(entry recipe.ModuleID, directory string) ([]workflowControl, string) {
			declared, refusal := controls(entry, directory)
			bound := make([]workflowControl, 0, len(declared))
			for _, control := range declared {
				name, kind, required, choices := control.Declared()
				label, media := control.Slot()
				wire := workflowControl{Name: name, Type: workflowControlType(kind), Required: required, Choices: choices, Label: label, Media: media}
				if defaultValue, step, rate, declared := control.Bounded(); declared {
					wire.Bounds = &workflowControlBounds{Default: defaultValue, Step: step, Rate: rate}
				}
				bound = append(bound, wire)
			}
			return bound, refusal
		},
		OutputContent: output,
	}
}

// storeGenerationWorkspace is the generation workspace over the store
// (professional GUI campaign, gui-generation-workspace): every active
// media recipe the store declares lists as a capability with the controls
// its request declares and the media inputs its recipe binds, executes
// through the same executors the recipe command uses, and publishes its
// output as a media artifact. Any launcher gets it with the store; no
// flag and no model name is involved.
type storeGenerationWorkspace struct {
	store   *overgodb.Store
	catalog generationCatalog
	limit   int
	mu      sync.Mutex
	memo    *discovery.Memo
}

// NewStoreGenerationWorkspace binds the workspace to the store and to the
// capability catalog the recipe command shares.
func NewStoreGenerationWorkspace(store *overgodb.Store, catalog generationCatalog, limit int) *storeGenerationWorkspace {
	return &storeGenerationWorkspace{store: store, catalog: catalog, limit: limit}
}

// generationTasks are the tasks a generation capability serves: the media
// outputs and the text answer to a question about an image.
var generationTasks = []recipe.Task{recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskSpeech, recipe.TaskVQA}

// generationRunKeyPrefix roots every generation run record's batch key.
const generationRunKeyPrefix = "generation/run/"

// recordedInput is the request document the runtime recorded for an
// executed output (the decoded request under the capability's input
// schema): the run cites it as its input, so a gallery re-opens the
// record beside the output and a page regenerates from it. A bare
// output outside the measured envelope records none.
func recordedInput(output any) (artifact.Content, bool) {
	measured, ok := output.(capabilityruntime.Measured)
	if !ok || measured.Input.Descriptor.ID == (artifact.ID{}) {
		return artifact.Content{}, false
	}
	return measured.Input, true
}

// WorkflowCapabilities lists one capability per active media recipe with
// present bytes, controls declared by the executor's request type, and a
// refusal on the entry when its request carries what a page cannot type.
func (workspace *storeGenerationWorkspace) WorkflowCapabilities(ctx context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || kind != workflowGeneration {
		return nil, nil
	}
	// Retain verified file identities; refresh activations on every read.
	workspace.mu.Lock()
	defer workspace.mu.Unlock()
	if err := workspace.store.Refresh(ctx); err != nil {
		return nil, err
	}
	if workspace.memo == nil {
		workspace.memo = discovery.LoadMemo(ctx, workspace.store)
	}
	entries, _, err := discovery.CapabilityCatalogForTasks(ctx, workspace.store, workspace.limit, workspace.memo, generationTasks...)
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
			if _, known := workspace.catalog.Execute[capability.Task]; !known {
				continue
			}
			selection, executionErr := modelrecipe.ResolveActiveExecution(ctx, workspace.store, entry.Model, capability.Task, modelrecipe.SessionWarm)
			program, refusal := selection.Program, ""
			if executionErr != nil {
				// An activation whose execution the store cannot resolve (a
				// recipe declaring no component session, a model without a
				// recorded extent) lists with the reason rather than failing
				// the workspace for every other activation.
				var err error
				if _, program, err = modelrecipe.ResolveActiveCapability(ctx, workspace.store, entry.Model, capability.Task); err != nil {
					return nil, fmt.Errorf("generation workspace: %s for %s: %w", capability.Task, entry.Model, err)
				}
				refusal = executionErr.Error()
			}
			definition := program.Definition()
			stages := program.Stages()
			if len(stages) == 0 {
				continue
			}
			directory, _ := workspace.executionPath(ctx, entry.Model, entry.Location)
			var declared []workflowControl
			if workspace.catalog.Controls != nil && refusal == "" {
				declared, refusal = workspace.catalog.Controls(stages[0].Module.ID, directory)
			}
			capabilities = append(capabilities, WorkflowCapability{
				Task: capability.Task, Recipe: definition.ID, Stages: stages,
				Inputs: definition.Inputs, Outputs: definition.Outputs, Controls: declared,
				Model: entry.Model, Name: filepath.Base(directory), Location: entry.Location, Refusal: refusal,
				execution: selection, directory: directory,
			})
		}
	}
	return capabilities, nil
}

// ExecuteWorkflow runs a listed capability against its model bytes with the
// execution its listing resolved and the request the page typed, publishes
// the output as a media artifact and a run record, and returns both as the
// operation's receipt.
func (workspace *storeGenerationWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, capability WorkflowCapability, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != workflowGeneration {
		return operation.Completion{}, errors.New("generation workspace: workflow is not admitted")
	}
	if capability.Refusal != "" {
		return operation.Completion{}, errors.New("generation workspace: " + capability.Refusal)
	}
	task, recipeID, listed := capability.Task, capability.Recipe, capability.execution.Activation.Definition
	execute, known := workspace.catalog.Execute[task]
	if !known || listed.ID != recipeID || listed.Task != task {
		return operation.Completion{}, fmt.Errorf("generation workspace: recipe %s is not an active %s capability", recipeID, task)
	}
	if workspace.catalog.OutputContent == nil {
		return operation.Completion{}, errors.New("generation workspace: the catalog publishes no output")
	}
	// The submission's sources (a prompt enhancement the page accepted) are
	// the run's inputs beside its request, so the original stays its source.
	sources := workflowSources(ctx)
	started := time.Now()
	begin, _ := processmeasure.Counter()
	output, err := execute(ctx, workspace.store, capability.directory, capability.execution, string(raw))
	end, _ := processmeasure.Counter()
	measured, _ := output.(capabilityruntime.Measured)
	var content artifact.Content
	if err == nil {
		content, err = workspace.catalog.OutputContent(capabilityruntime.Unwrap(output))
	}
	if err != nil {
		completion, failed := failWorkflow(ctx, workspace.store, recipeID, sources, generationFailure, err)
		workspace.observe(ctx, capability.Model, recipeID, completion.Run, task, started, end-begin, measured, failed)
		return completion, failed
	}
	// The recorded request is the run's input: the record a gallery opens.
	inputs := slices.Clone(sources)
	request, recorded := recordedInput(output)
	if recorded {
		inputs = append(inputs, request.Descriptor.ID)
	}
	run, err := runrecord.NewRun(recipeID, runrecord.OutcomeSucceeded, inputs, []artifact.ID{content.Descriptor.ID}, "")
	if err != nil {
		return operation.Completion{}, err
	}
	batch, err := run.Batch(generationRunKeyPrefix + run.ID.String())
	if err != nil {
		return operation.Completion{}, err
	}
	if recorded {
		batch.Contents = append(batch.Contents, request)
	}
	batch.Contents = append(batch.Contents, content)
	if _, err := artifact.Publish(ctx, workspace.store, batch); err != nil {
		return operation.Completion{}, err
	}
	workspace.observe(ctx, capability.Model, recipeID, run.ID, task, started, end-begin, measured, nil)
	reporter.Publishing()
	return operation.Completion{Run: run.ID, Outputs: []artifact.ID{content.Descriptor.ID}}, nil
}

// generationFailure is the failure code a failed media execution records.
const generationFailure = "generation_failed"

// observe records one media execution as the serving attempt the scorecard
// counts. The output is already made or refused, so a record the store
// refuses fails nothing.
func (workspace *storeGenerationWorkspace) observe(ctx context.Context, model, recipeID, run artifact.ID, task recipe.Task, started time.Time, wall time.Duration, measured capabilityruntime.Measured, cause error) {
	_ = servingattempt.RecordWholeOutput(context.WithoutCancel(ctx), workspace.store, servingattempt.WholeOutput{
		Model: model, Recipe: recipeID, Run: run, Task: task, Started: started, Wall: wall,
		PeakDeviceBytes: measured.PeakDeviceBytes, Cause: cause,
	})
}

// executionPath names the model directory an executor reads: every media
// capability resolves a repository directory (config beside weights), so
// the recorded directory location serves when the store holds one, else
// the directory holding the present file the catalog resolved.
func (workspace *storeGenerationWorkspace) executionPath(ctx context.Context, model artifact.ID, location string) (string, error) {
	if path, err := artifact.AvailablePath(ctx, workspace.store, model, artifact.LocationDirectory); err == nil {
		return path, nil
	}
	if location == "" {
		return "", fmt.Errorf("generation workspace: model %s has no bytes on disk", model)
	}
	return filepath.Dir(location), nil
}
