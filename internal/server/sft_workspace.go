package server

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/trainingworkflow"
)

// SFTTrainer is the launcher's supervised fine-tuner of the served model:
// the model's artifact and GGUF file, and Render, which renders a conversation
// through the model's own chat template. A run trains on the host processor,
// as the workbench's other training does.
type SFTTrainer struct {
	Model  artifact.ID
	File   string
	Render trainingworkflow.ConversationRenderer
}

// sftRecipeRuntime is the program a conversation run's recipe identity
// derives from.
const sftRecipeRuntime = "conversation-sft/v1"

// sftDatasetFormat is the catalog format of a conversation dataset the page
// offers: one JSON file per conversation.
const sftDatasetFormat = "json"

// sftNote tells the page how a run learns from its dataset.
const sftNote = "Learns every token of each conversation as the model's chat template renders it, the user's turns and the template's own markers as well as the replies: the trainer has no loss mask yet."

// SFTWorkspace offers the Train tab a fine-tuning of the served model on a
// conversation dataset: each conversation is rendered through the model's
// chat template, a few are held out, the model trains on the rest, and the
// trained model and its held-out verdict are recorded as one run.
type SFTWorkspace struct {
	store   *overgodb.Store
	roots   dataroot.Roots
	trainer SFTTrainer
}

// NewSFTWorkspace binds the workspace to the store, the data roots it writes
// under and the trainer.
func NewSFTWorkspace(store *overgodb.Store, roots dataroot.Roots, trainer SFTTrainer) (*SFTWorkspace, error) {
	if store == nil || !trainer.Model.Valid() || trainer.File == "" || trainer.Render == nil || roots.Checkpoints == "" || roots.Datasets == "" {
		return nil, errors.New("sft workspace: a store, the served model and its renderer, and the checkpoints and datasets roots are required")
	}
	return &SFTWorkspace{store: store, roots: roots, trainer: trainer}, nil
}

// WorkflowCapabilities lists the one fine-tuning the Train tab offers: the
// served model on a conversation dataset.
func (workspace *SFTWorkspace) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || kind != WorkflowTraining {
		return nil, nil
	}
	program, err := ggufProgram(sftRecipeRuntime, workspace.trainer.Model)
	if err != nil {
		return nil, err
	}
	return []WorkflowCapability{{
		Task: recipe.TaskTraining, Recipe: program.ID, Model: workspace.trainer.Model,
		Name: filepath.Base(workspace.trainer.File) + ", on conversations", Location: workspace.trainer.File,
		Host:    trainsOnHost,
		Note:    sftNote,
		Stages:  []recipe.Stage{{Node: recipe.Node{ID: "train", Module: sftRecipeRuntime}}},
		Outputs: []recipe.Output{{Name: "model", Data: recipe.DataArtifact, Source: recipe.Endpoint{Node: "train", Port: "model"}}},
		Controls: []workflowControl{
			{Name: "dataset", Type: workflowControlDataset, Required: true, Label: "conversations", Format: sftDatasetFormat},
			{Name: "steps", Type: workflowControlInteger, Required: true},
			{Name: "output", Type: workflowControlOutput, Required: true},
		},
	}}, nil
}

type sftInput struct {
	Dataset artifact.ID `json:"dataset"`
	Steps   int         `json:"steps"`
	Output  string      `json:"output"`
}

// ExecuteWorkflow fine-tunes the served model on the dataset and records the
// run: the model and the conversations in, the trained model and the verdict
// that judged it out. A run that fails or is cancelled records its outcome.
func (workspace *SFTWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, capability WorkflowCapability, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != WorkflowTraining || capability.Task != recipe.TaskTraining {
		return operation.Completion{}, errors.New("sft workspace: workflow is not admitted")
	}
	program, err := ggufProgram(sftRecipeRuntime, workspace.trainer.Model)
	if err != nil {
		return operation.Completion{}, err
	}
	if capability.Recipe != program.ID {
		return operation.Completion{}, errors.New("sft workspace: workflow is not admitted")
	}
	var input sftInput
	if err := strictjson.DecodeBytes(raw, &input); err != nil {
		return operation.Completion{}, err
	}
	inputs := []artifact.ID{workspace.trainer.Model, input.Dataset}
	if input.Dataset.Kind() != artifact.KindDataset || input.Steps <= 0 {
		return operation.Completion{}, errors.New("sft workspace: a conversation dataset and a positive number of steps are required")
	}
	output, err := outputDirectory(workspace.roots.Checkpoints, input.Output)
	if err != nil {
		return failWorkflow(ctx, workspace.store, program.ID, inputs, "training_failed", err)
	}
	total := uint64(input.Steps)
	reporter.Progress(0, &total)
	result, err := trainingworkflow.RunConversationSFT(ctx, workspace.store, trainingworkflow.ConversationSFT{
		Dataset: input.Dataset, Render: workspace.trainer.Render, ModelDirectory: workspace.trainer.File,
		OutputDirectory: output, WorkDirectory: filepath.Join(workspace.roots.Datasets, "sft"),
		Steps: input.Steps,
		Step: func(step, _ int, loss float64) {
			reporter.Progress(uint64(step), &total)
			reporter.Metric(operation.Metric{Name: "loss", Value: loss})
		},
	})
	if err != nil {
		return failWorkflow(ctx, workspace.store, program.ID, inputs, "training_failed", err)
	}
	reporter.Publishing()
	outputs := []artifact.ID{result.Trained, result.Verdict}
	run, err := runrecord.NewRun(program.ID, runrecord.OutcomeSucceeded, inputs, outputs, "")
	if err != nil {
		return operation.Completion{}, err
	}
	batch, err := run.Batch("sft/run/" + run.ID.String())
	if err != nil {
		return operation.Completion{}, err
	}
	batch.Artifacts = append(batch.Artifacts, program)
	if _, err := artifact.CommitBatch(ctx, workspace.store, batch); err != nil {
		return operation.Completion{}, err
	}
	return operation.Completion{Run: run.ID, Outputs: outputs}, nil
}

// PreviewWorkflow pages the input's conversations as a run reads them: each
// rendered through the model's chat template and split into tokens, before any
// run.
func (workspace *SFTWorkspace) PreviewWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, position, limit int) (any, error) {
	if workspace == nil || kind != WorkflowTraining || task != recipe.TaskTraining {
		return nil, errors.New("sft workspace: workflow is not admitted")
	}
	program, err := ggufProgram(sftRecipeRuntime, workspace.trainer.Model)
	if err != nil {
		return nil, err
	}
	if recipeID != program.ID {
		return nil, errors.New("sft workspace: workflow is not admitted")
	}
	var input struct {
		Dataset artifact.ID `json:"dataset"`
	}
	if err := json.Unmarshal(raw, &input); err != nil || input.Dataset.Kind() != artifact.KindDataset {
		return nil, errors.Join(err, errors.New("sft workspace: preview needs a conversation dataset"))
	}
	return trainingworkflow.PreviewConversations(ctx, workspace.store, input.Dataset, workspace.trainer.Render, workspace.trainer.File, position, limit)
}

// outputDirectory is the directory a run named name writes under root: the
// name is one safe path component.
func outputDirectory(root, name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" || name == "." || name == ".." || filepath.Base(name) != name || strings.ContainsAny(name, "\x00\r\n") {
		return "", errors.New("training workspace: output must be one safe path component")
	}
	return filepath.Join(root, name), nil
}
