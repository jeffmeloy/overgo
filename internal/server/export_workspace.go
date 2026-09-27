package server

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

// GGUFExporter writes a catalogued GGUF model in a distribution form: Types
// are the types it writes, and Write writes the model at source into a new
// directory named by the output under root, returning the written file.
type GGUFExporter struct {
	Types []string
	Write func(source, root, output, typeName string) (string, error)
}

// ExportWorkspace offers each catalogued GGUF model to the exporter the
// launcher binds, and records an export as a run whose input is the model
// and whose output is the written file at its location.
type ExportWorkspace struct {
	store    *overgodb.Store
	root     string
	limit    int
	exporter GGUFExporter
	mu       sync.Mutex
	memo     *discovery.Memo
}

// NewExportWorkspace binds the workspace to the store, the directory exports
// are written under, the catalog bound and the exporter.
func NewExportWorkspace(store *overgodb.Store, root string, limit int, exporter GGUFExporter) *ExportWorkspace {
	return &ExportWorkspace{store: store, root: root, limit: limit, exporter: exporter}
}

// exportRecipeRuntime names the export program a capability's recipe
// identity derives from.
const exportRecipeRuntime = "gguf-export/v1"

// exportRecipe is the export program for one model: the identity of its
// declaration, and the declaration's size a run's batch records it with.
func exportRecipe(model artifact.ID) (artifact.Descriptor, error) {
	declaration, err := json.Marshal(struct {
		Runtime string      `json:"runtime"`
		Model   artifact.ID `json:"model"`
	}{Runtime: exportRecipeRuntime, Model: model})
	if err != nil {
		return artifact.Descriptor{}, err
	}
	id, err := artifact.IdentifyBytes(artifact.KindRecipe, declaration)
	return artifact.Descriptor{ID: id, Size: uint64(len(declaration))}, err
}

// WorkflowCapabilities lists one export per catalogued inference model whose
// GGUF file is present.
func (workspace *ExportWorkspace) WorkflowCapabilities(ctx context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || kind != WorkflowExport || workspace.exporter.Write == nil {
		return nil, nil
	}
	workspace.mu.Lock()
	defer workspace.mu.Unlock()
	if err := workspace.store.Refresh(ctx); err != nil {
		return nil, err
	}
	if workspace.memo == nil {
		workspace.memo = discovery.LoadMemo(ctx, workspace.store)
	}
	entries, _, err := discovery.CapabilityCatalogForTasks(ctx, workspace.store, workspace.limit, workspace.memo, recipe.TaskInference)
	if err != nil {
		return nil, err
	}
	var capabilities []WorkflowCapability
	for _, entry := range entries {
		if !entry.Present || !strings.EqualFold(filepath.Ext(entry.Location), ".gguf") {
			continue
		}
		program, err := exportRecipe(entry.Model)
		if err != nil {
			return nil, err
		}
		capabilities = append(capabilities, WorkflowCapability{
			Task: recipe.TaskExport, Recipe: program.ID, Model: entry.Model, Name: filepath.Base(entry.Location), Location: entry.Location,
			Stages:  []recipe.Stage{{Node: recipe.Node{ID: "quantize", Module: exportRecipeRuntime}}},
			Outputs: []recipe.Output{{Name: "model", Data: recipe.DataArtifact, Source: recipe.Endpoint{Node: "quantize", Port: "model"}}},
			Controls: []WorkflowControl{
				{Name: "type", Type: WorkflowControlText, Required: true, Choices: workspace.exporter.Types},
				{Name: "output", Type: WorkflowControlOutput, Required: true},
			},
		})
	}
	return capabilities, nil
}

type exportInput struct {
	Type   string `json:"type"`
	Output string `json:"output"`
}

// ExecuteWorkflow writes the capability's model through the exporter and
// records the run with the written file's identity and location.
func (workspace *ExportWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != WorkflowExport || task != recipe.TaskExport {
		return operation.Completion{}, errors.New("export workspace: workflow is not admitted")
	}
	capabilities, err := workspace.WorkflowCapabilities(ctx, kind)
	if err != nil {
		return operation.Completion{}, err
	}
	capability, err := selectWorkflowCapability(capabilities, task, recipeID)
	if err != nil {
		return operation.Completion{}, err
	}
	var input exportInput
	if err := strictjson.DecodeBytes(raw, &input); err != nil {
		return operation.Completion{}, err
	}
	file, err := workspace.exporter.Write(capability.Location, workspace.root, input.Output, input.Type)
	if err != nil {
		return operation.Completion{}, err
	}
	reporter.Publishing()
	completion, err := workspace.record(ctx, capability.Model, file)
	if err != nil {
		return operation.Completion{}, errors.Join(err, os.RemoveAll(filepath.Dir(file)))
	}
	total := uint64(1)
	reporter.Progress(total, &total)
	return completion, nil
}

// record commits the export's run: the model in, the written file out, at
// its location.
func (workspace *ExportWorkspace) record(ctx context.Context, model artifact.ID, file string) (operation.Completion, error) {
	program, err := exportRecipe(model)
	if err != nil {
		return operation.Completion{}, err
	}
	written, err := os.Open(file)
	if err != nil {
		return operation.Completion{}, err
	}
	exported, size, err := artifact.Identify(artifact.KindModel, written)
	if err := errors.Join(err, written.Close()); err != nil {
		return operation.Completion{}, err
	}
	run, err := runrecord.NewRun(program.ID, runrecord.OutcomeSucceeded, []artifact.ID{model}, []artifact.ID{exported}, "")
	if err != nil {
		return operation.Completion{}, err
	}
	batch, err := run.Batch("export/run/" + run.ID.String())
	if err != nil {
		return operation.Completion{}, err
	}
	batch.Artifacts = append(batch.Artifacts, program, artifact.Descriptor{ID: exported, Size: size})
	batch.Locations = append(batch.Locations, artifact.LocationEvent{
		Location: artifact.Location{Artifact: exported, Kind: artifact.LocationFile, Value: file}, Action: artifact.LocationAdd,
	})
	if _, err := artifact.CommitBatch(ctx, workspace.store, batch); err != nil {
		return operation.Completion{}, err
	}
	return operation.Completion{Run: run.ID, Outputs: []artifact.ID{exported}}, nil
}
