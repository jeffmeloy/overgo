package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
	"overgo/internal/workflowrecipe"
)

type TrainingWorkspace struct {
	store   artifact.Repository
	roots   dataroot.Roots
	program recipe.Program
}

func NewTrainingWorkspace(ctx context.Context, store artifact.Repository, roots dataroot.Roots, model artifact.ID) (*TrainingWorkspace, error) {
	if ctx == nil || store == nil || model.Kind() != artifact.KindModel || roots.Checkpoints == "" {
		return nil, errors.New("training workspace: invalid store, roots, or model")
	}
	_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, model, recipe.TaskTraining)
	if err != nil {
		return nil, err
	}
	if err := validateDPOProgram(program); err != nil {
		return nil, err
	}
	return &TrainingWorkspace{store: store, roots: roots, program: program}, nil
}

func (workspace *TrainingWorkspace) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || kind != WorkflowTraining {
		return nil, nil
	}
	definition := workspace.program.Definition()
	return []WorkflowCapability{{
		Task: definition.Task, Recipe: definition.ID, Stages: workspace.program.Stages(),
		Inputs: definition.Inputs, Outputs: definition.Outputs,
		Controls: []WorkflowControl{
			{Name: "dataset", Type: WorkflowControlDataset, Required: true},
			{Name: "resume", Type: WorkflowControlCheckpoint},
			{Name: "output", Type: WorkflowControlOutput, Required: true},
			{Name: "steps", Type: WorkflowControlInteger, Required: true},
			{Name: "learning_rate", Type: WorkflowControlNumber, Required: true},
			{Name: "momentum", Type: WorkflowControlNumber, Required: true},
			{Name: "dpo_scale", Type: WorkflowControlNumber, Required: true},
		},
	}}, nil
}

type dpoWorkflowInput struct {
	Dataset      artifact.ID `json:"dataset"`
	Resume       artifact.ID `json:"resume,omitempty"`
	Output       string      `json:"output"`
	Steps        int         `json:"steps"`
	LearningRate float64     `json:"learning_rate"`
	Momentum     float64     `json:"momentum"`
	DPOScale     float64     `json:"dpo_scale"`
}

func (workspace *TrainingWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != WorkflowTraining || task != recipe.TaskTraining || recipeID != workspace.program.Definition().ID {
		return operation.Completion{}, errors.New("training workspace: workflow is not admitted")
	}
	var input dpoWorkflowInput
	if err := strictjson.DecodeBytes(raw, &input); err != nil {
		return operation.Completion{}, err
	}
	definition := workspace.program.Definition()
	policy, ok := definition.Dependency(recipe.DependencyModel, 0)
	if !ok {
		return operation.Completion{}, errors.New("training workspace: policy dependency absent")
	}
	reference, ok := definition.Dependency(recipe.DependencyModel, 1)
	if !ok {
		return operation.Completion{}, errors.New("training workspace: reference dependency absent")
	}
	policyDirectory, err := workspace.directory(ctx, policy)
	if err != nil {
		return workspace.fail(ctx, definition.ID, []artifact.ID{policy, reference, input.Dataset}, err)
	}
	referenceDirectory, err := workspace.directory(ctx, reference)
	if err != nil {
		return workspace.fail(ctx, definition.ID, []artifact.ID{policy, reference, input.Dataset}, err)
	}
	datasetPath, err := workspace.file(ctx, input.Dataset)
	if err != nil {
		return workspace.fail(ctx, definition.ID, []artifact.ID{policy, reference, input.Dataset}, err)
	}
	resumeDirectory := ""
	inputs := []artifact.ID{policy, reference, input.Dataset}
	if input.Resume.Valid() {
		resumeDirectory, err = workspace.directory(ctx, input.Resume)
		if err != nil {
			return workspace.fail(ctx, definition.ID, inputs, err)
		}
		inputs = append(inputs, input.Resume)
	}
	output, err := workspace.outputPath(input.Output)
	if err != nil {
		return workspace.fail(ctx, definition.ID, inputs, err)
	}
	total := uint64(input.Steps)
	reporter.Progress(0, &total)
	result, executeErr := trainingworkflow.Execute(ctx, trainingworkflow.Request{
		Recipe:         recipeID,
		ModelDirectory: policyDirectory, ReferenceDirectory: referenceDirectory,
		DatasetPath: datasetPath, OutputDirectory: output, ResumeDirectory: resumeDirectory,
		Steps: input.Steps, LearningRate: input.LearningRate, Momentum: input.Momentum,
		DPOScale: input.DPOScale, Host: true,
	})
	if executeErr != nil {
		return workspace.fail(ctx, definition.ID, inputs, executeErr)
	}
	reporter.Progress(total, &total)
	reporter.Publishing()
	return workspace.publish(ctx, definition.ID, inputs, result.Checkpoint, output)
}

func validateDPOProgram(program recipe.Program) error {
	stages := program.Stages()
	modules := make([]recipe.ModuleID, len(stages))
	for index, stage := range stages {
		modules[index] = stage.Module.ID
	}
	want := []recipe.ModuleID{
		workflowrecipe.ModuleBatchPreference, workflowrecipe.ModuleScorePolicy,
		workflowrecipe.ModuleScoreReference, workflowrecipe.ModuleDPOObjective,
		workflowrecipe.ModuleBackward, workflowrecipe.ModuleOptimize,
	}
	if program.Definition().Task != recipe.TaskTraining || !slices.Equal(modules, want) {
		return fmt.Errorf("training workspace: active recipe stages %v are not native DPO", modules)
	}
	return nil
}

func (workspace *TrainingWorkspace) directory(ctx context.Context, id artifact.ID) (string, error) {
	locations, err := workspace.store.Locations(ctx, id)
	if err != nil {
		return "", err
	}
	for _, location := range locations {
		path := location.Value
		if location.Kind == artifact.LocationFile {
			path = filepath.Dir(path)
		} else if location.Kind != artifact.LocationDirectory {
			continue
		}
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			return path, nil
		}
	}
	return "", fmt.Errorf("training workspace: %s has no available directory", id)
}

func (workspace *TrainingWorkspace) file(ctx context.Context, id artifact.ID) (string, error) {
	if id.Kind() != artifact.KindDataset {
		return "", errors.New("training workspace: dataset identity required")
	}
	locations, err := workspace.store.Locations(ctx, id)
	if err != nil {
		return "", err
	}
	for _, location := range locations {
		if location.Kind == artifact.LocationFile {
			if info, err := os.Stat(location.Value); err == nil && !info.IsDir() {
				return location.Value, nil
			}
		}
	}
	return "", fmt.Errorf("training workspace: %s has no available file", id)
}

func (workspace *TrainingWorkspace) outputPath(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" || name == "." || name == ".." || filepath.Base(name) != name || strings.ContainsAny(name, "\x00\r\n") {
		return "", errors.New("training workspace: output must be one safe path component")
	}
	return filepath.Join(workspace.roots.Checkpoints, name), nil
}

func (workspace *TrainingWorkspace) publish(ctx context.Context, recipeID artifact.ID, inputs []artifact.ID, checkpoint trainingprogram.Checkpoint, path string) (operation.Completion, error) {
	data, err := checkpoint.Marshal()
	if err != nil {
		return operation.Completion{}, err
	}
	run, err := runrecord.NewRun(recipeID, runrecord.OutcomeSucceeded, inputs, []artifact.ID{checkpoint.ID()}, "")
	if err != nil {
		return operation.Completion{}, err
	}
	batch, err := run.Batch("training/run/" + run.ID.String())
	if err != nil {
		return operation.Completion{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: checkpoint.ID(), Size: uint64(len(data))})
	batch.Locations = append(batch.Locations, artifact.LocationEvent{
		Location: artifact.Location{Artifact: checkpoint.ID(), Kind: artifact.LocationDirectory, Value: path},
		Action:   artifact.LocationAdd,
	})
	if _, err := artifact.CommitBatch(ctx, workspace.store, batch); err != nil {
		return operation.Completion{}, err
	}
	return operation.Completion{Run: run.ID, Outputs: []artifact.ID{checkpoint.ID()}}, nil
}

func (workspace *TrainingWorkspace) fail(ctx context.Context, recipeID artifact.ID, inputs []artifact.ID, cause error) (operation.Completion, error) {
	outcome, failure := runrecord.OutcomeFailed, "training_failed"
	if errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) {
		outcome, failure = runrecord.OutcomeCancelled, ""
	}
	run, err := runrecord.NewRun(recipeID, outcome, inputs, nil, failure)
	if err == nil {
		batch, batchErr := run.Batch("training/run/" + run.ID.String())
		commitContext := ctx
		if ctx.Err() != nil {
			commitContext = context.WithoutCancel(ctx)
		}
		if batchErr == nil {
			_, batchErr = artifact.CommitBatch(commitContext, workspace.store, batch)
		}
		err = errors.Join(err, batchErr)
	}
	return operation.Completion{Run: run.ID}, errors.Join(cause, err)
}
