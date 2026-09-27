package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/discovery"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

// GGUFWriters are the launcher's GGUF writers: Export writes the model at
// source quantized to a type (one of Types) with an Ollama Modelfile, and
// Merge writes base + sum(scale * (fine-tuned - base)); each writes into a
// new directory named by the output under root and returns the written file.
type GGUFWriters struct {
	Types  []string
	Export func(source, root, output, typeName string) (string, error)
	Merge  func(base string, tuned []string, scales []float32, root, output string) (string, error)
}

// GGUFWorkspace offers every catalogued GGUF model to the launcher's GGUF
// writers -- as an export in the Export tab and as the base of a merge in
// Model Builder -- and records each as a run with its models in and the
// written file out at its location.
type GGUFWorkspace struct {
	store   *overgodb.Store
	root    string
	limit   int
	writers GGUFWriters
	// commit is the executable's clean source revision; empty, a run binds
	// no source, environment or measured wall.
	commit      string
	environment runrecord.Environment
	mu          sync.Mutex
	memo        *discovery.Memo
}

// NewGGUFWorkspace binds the workspace to the store, the directory it writes
// under, the catalog bound, the writers and the executable's source commit.
func NewGGUFWorkspace(store *overgodb.Store, root string, limit int, writers GGUFWriters, commit string) *GGUFWorkspace {
	return &GGUFWorkspace{store: store, root: root, limit: limit, writers: writers, commit: commit}
}

// The programs a capability's recipe identity derives from.
const (
	exportRecipeRuntime      = "gguf-export/v1"
	mergeRecipeRuntime       = "gguf-merge/v1"
	adapterFoldRecipeRuntime = "gguf-adapter-fold/v1"
)

// ggufProgram is a program's recipe for one model (none for the adapter
// fold): the identity of its declaration and the declaration's size a run's
// batch records it with.
func ggufProgram(runtime string, model artifact.ID) (artifact.Descriptor, error) {
	declaration, err := json.Marshal(struct {
		Runtime string      `json:"runtime"`
		Model   artifact.ID `json:"model,omitzero"`
	}{Runtime: runtime, Model: model})
	if err != nil {
		return artifact.Descriptor{}, err
	}
	id, err := artifact.IdentifyBytes(artifact.KindRecipe, declaration)
	return artifact.Descriptor{ID: id, Size: uint64(len(declaration))}, err
}

// catalogued is the store's inference models whose GGUF file is present.
func (workspace *GGUFWorkspace) catalogued(ctx context.Context) ([]discovery.CatalogEntry, error) {
	workspace.mu.Lock()
	defer workspace.mu.Unlock()
	if err := workspace.store.Refresh(ctx); err != nil {
		return nil, err
	}
	if workspace.memo == nil {
		workspace.memo = discovery.LoadMemo(ctx, workspace.store)
	}
	entries, _, err := discovery.CapabilityCatalogForTasks(ctx, workspace.store, workspace.limit, workspace.memo, recipe.TaskInference)
	var present []discovery.CatalogEntry
	for _, entry := range entries {
		if entry.Present && strings.EqualFold(filepath.Ext(entry.Location), ".gguf") {
			present = append(present, entry)
		}
	}
	return present, err
}

// WorkflowCapabilities lists an export per catalogued GGUF model in the
// Export tab, and in Model Builder a merge onto each as the base with the
// others as the fine-tuned choice, beside the adapter fold with its refusal.
func (workspace *GGUFWorkspace) WorkflowCapabilities(ctx context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || (kind != WorkflowExport && kind != WorkflowModelBuild) {
		return nil, nil
	}
	entries, err := workspace.catalogued(ctx)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, filepath.Base(entry.Location))
	}
	var capabilities []WorkflowCapability
	for _, entry := range entries {
		capability, err := workspace.capability(kind, entry, names)
		if err != nil {
			return nil, err
		}
		if capability.Controls != nil {
			capabilities = append(capabilities, capability)
		}
	}
	if kind == WorkflowModelBuild {
		fold, err := ggufProgram(adapterFoldRecipeRuntime, artifact.ID{})
		if err != nil {
			return nil, err
		}
		capabilities = append(capabilities, WorkflowCapability{Task: recipe.TaskMerge, Recipe: fold.ID, Name: "an adapter into its base",
			Stages:   []recipe.Stage{{Node: recipe.Node{ID: "fold", Module: adapterFoldRecipeRuntime}}},
			Controls: []WorkflowControl{}, Refusal: "folding a trained GGUF adapter into its base has no implementation yet"})
	}
	return capabilities, nil
}

// capability is one model's export, or its merge as the base when another
// catalogued model can be merged into it.
func (workspace *GGUFWorkspace) capability(kind WorkflowKind, entry discovery.CatalogEntry, names []string) (WorkflowCapability, error) {
	name := filepath.Base(entry.Location)
	capability := WorkflowCapability{Model: entry.Model, Name: name, Location: entry.Location,
		Outputs: []recipe.Output{{Name: "model", Data: recipe.DataArtifact, Source: recipe.Endpoint{Node: "write", Port: "model"}}}}
	runtime := exportRecipeRuntime
	switch {
	case kind == WorkflowExport && workspace.writers.Export != nil:
		capability.Task = recipe.TaskExport
		capability.Controls = []WorkflowControl{{Name: "type", Type: WorkflowControlText, Required: true, Choices: workspace.writers.Types}}
	case kind == WorkflowModelBuild && workspace.writers.Merge != nil && len(names) > 1:
		runtime, capability.Task = mergeRecipeRuntime, recipe.TaskMerge
		others := make([]string, 0, len(names)-1)
		for _, other := range names {
			if other != name {
				others = append(others, other)
			}
		}
		capability.Controls = []WorkflowControl{
			{Name: "fine_tuned", Type: WorkflowControlText, Required: true, Choices: others},
			{Name: "scale", Type: WorkflowControlNumber, Required: true},
		}
	default:
		return WorkflowCapability{}, nil
	}
	capability.Controls = append(capability.Controls, WorkflowControl{Name: "output", Type: WorkflowControlOutput, Required: true})
	capability.Stages = []recipe.Stage{{Node: recipe.Node{ID: "write", Module: recipe.ModuleID(runtime)}}}
	program, err := ggufProgram(runtime, entry.Model)
	capability.Recipe = program.ID
	return capability, err
}

type ggufWriteInput struct {
	Type      string  `json:"type,omitzero"`
	FineTuned string  `json:"fine_tuned,omitzero"`
	Scale     float32 `json:"scale,omitzero"`
	Output    string  `json:"output"`
}

// ExecuteWorkflow writes the capability's model through its writer and
// records the run with the written file's identity and location.
func (workspace *GGUFWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil {
		return operation.Completion{}, errors.New("gguf workspace: workflow is not admitted")
	}
	capabilities, err := workspace.WorkflowCapabilities(ctx, kind)
	if err != nil {
		return operation.Completion{}, err
	}
	capability, err := selectWorkflowCapability(capabilities, task, recipeID)
	if err != nil {
		return operation.Completion{}, err
	}
	if capability.Refusal != "" {
		return operation.Completion{}, errors.New("gguf workspace: " + capability.Refusal)
	}
	var input ggufWriteInput
	if err := strictjson.DecodeBytes(raw, &input); err != nil {
		return operation.Completion{}, err
	}
	inputs := []artifact.ID{capability.Model}
	runtime := mergeRecipeRuntime
	var file string
	stopwatch := processmeasure.NewStopwatch()
	if task == recipe.TaskExport {
		runtime = exportRecipeRuntime
		file, err = workspace.writers.Export(capability.Location, workspace.root, input.Output, input.Type)
	} else {
		var tuned discovery.CatalogEntry
		if tuned, err = workspace.fineTuned(ctx, input.FineTuned); err == nil && !checked.Finite32(input.Scale) {
			err = errors.New("gguf workspace: the scale must be finite")
		}
		if err == nil {
			inputs = append(inputs, tuned.Model)
			file, err = workspace.writers.Merge(capability.Location, []string{tuned.Location}, []float32{input.Scale}, workspace.root, input.Output)
		}
	}
	if err != nil {
		return operation.Completion{}, err
	}
	reporter.Publishing()
	measured, err := stopwatch.Elapsed()
	if err != nil {
		return operation.Completion{}, errors.Join(err, os.RemoveAll(filepath.Dir(file)))
	}
	program, err := ggufProgram(runtime, capability.Model)
	if err != nil {
		return operation.Completion{}, errors.Join(err, os.RemoveAll(filepath.Dir(file)))
	}
	completion, err := workspace.record(ctx, program, inputs, file, measured)
	if err != nil {
		return operation.Completion{}, errors.Join(err, os.RemoveAll(filepath.Dir(file)))
	}
	total := uint64(1)
	reporter.Progress(total, &total)
	return completion, nil
}

// fineTuned is the one catalogued GGUF model with the file name.
func (workspace *GGUFWorkspace) fineTuned(ctx context.Context, name string) (discovery.CatalogEntry, error) {
	entries, err := workspace.catalogued(ctx)
	if err != nil {
		return discovery.CatalogEntry{}, err
	}
	var matched []discovery.CatalogEntry
	for _, entry := range entries {
		if filepath.Base(entry.Location) == name {
			matched = append(matched, entry)
		}
	}
	if len(matched) != 1 {
		return discovery.CatalogEntry{}, fmt.Errorf("gguf workspace: %d catalogued models are named %q; one must be", len(matched), name)
	}
	return matched[0], nil
}

// hostEnvironment is this process's execution platform, published once as
// the fact its runs cite.
func (workspace *GGUFWorkspace) hostEnvironment(ctx context.Context) (runrecord.Environment, error) {
	workspace.mu.Lock()
	defer workspace.mu.Unlock()
	if workspace.environment.ID.Valid() {
		return workspace.environment, nil
	}
	environment, err := runrecord.CurrentEnvironment("host", "go")
	if err != nil {
		return runrecord.Environment{}, err
	}
	batch, err := environment.Batch("gguf/environment/" + environment.ID.String())
	if err == nil {
		_, err = artifact.Publish(ctx, workspace.store, batch)
	}
	if err != nil {
		return runrecord.Environment{}, err
	}
	workspace.environment = environment
	return environment, nil
}

// record commits the run: the models in, the written file out at its
// location, under the program's recipe.
func (workspace *GGUFWorkspace) record(ctx context.Context, program artifact.Descriptor, inputs []artifact.ID, file string, measured uint64) (operation.Completion, error) {
	written, err := os.Open(file)
	if err != nil {
		return operation.Completion{}, err
	}
	output, size, err := artifact.Identify(artifact.KindModel, written)
	if err := errors.Join(err, written.Close()); err != nil {
		return operation.Completion{}, err
	}
	run, err := runrecord.NewRun(program.ID, runrecord.OutcomeSucceeded, inputs, []artifact.ID{output}, "")
	if workspace.commit != "" {
		// A known source binds the run to it, to this host and to its measured wall.
		var environment runrecord.Environment
		if environment, err = workspace.hostEnvironment(ctx); err == nil {
			run, err = runrecord.NewBoundRun(program.ID, runrecord.OutcomeSucceeded, inputs, []artifact.ID{output}, "", workspace.commit, environment.ID, measured, nil)
		}
	}
	if err != nil {
		return operation.Completion{}, err
	}
	batch, err := run.Batch("gguf/run/" + run.ID.String())
	if err != nil {
		return operation.Completion{}, err
	}
	batch.Artifacts = append(batch.Artifacts, program, artifact.Descriptor{ID: output, Size: size})
	batch.Locations = append(batch.Locations, artifact.LocationEvent{
		Location: artifact.Location{Artifact: output, Kind: artifact.LocationFile, Value: file}, Action: artifact.LocationAdd,
	})
	if _, err := artifact.CommitBatch(ctx, workspace.store, batch); err != nil {
		return operation.Completion{}, err
	}
	return operation.Completion{Run: run.ID, Outputs: []artifact.ID{output}}, nil
}
