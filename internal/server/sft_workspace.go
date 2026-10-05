package server

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/fsatomic"
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

const (
	// sftRecipeRuntime and sftLoRARuntime are the programs a conversation run's
	// recipe identity derives from, one for each way of fine-tuning.
	sftRecipeRuntime = "conversation-sft/v1"
	sftLoRARuntime   = "conversation-lora/v1"
	// sftLoRADefaultRank is the rank the page offers a LoRA adapter, the one
	// PEFT's and LlamaFactory's examples use.
	sftLoRADefaultRank = 8
	// sftCheckpointsKept is how many of the checkpoints a run saves along the way
	// the workbench keeps: the newest.
	sftCheckpointsKept = 3
	// sftDatasetFormat is the catalog format of a conversation dataset the page
	// offers: one JSON file per conversation.
	sftDatasetFormat = "json"
	// sftNote and sftLoRANote tell the page how a run learns from its dataset.
	sftNote     = "Learns the replies of each conversation as the model's chat template renders them, and only reads the user's turns and the template's own markers; train on prompts: true learns every token, and average weights: true publishes a moving average of the weights."
	sftLoRANote = "Trains a low-rank adapter over the frozen model rather than the model, and saves it as a Hugging Face PEFT adapter in the output's adapter directory. Like full fine-tuning it learns the replies of each conversation and only reads the rest; train on prompts: true learns every token, and average weights: true publishes a moving average of the adapter's factors."
)

// sftMethod is one way the workspace fine-tunes the served model on
// conversations: its program, how the page names and describes it, and whether
// it trains a LoRA adapter, which takes a rank.
type sftMethod struct {
	runtime, name, note string
	lora                bool
}

var sftMethods = []sftMethod{
	{runtime: sftRecipeRuntime, name: ", on conversations", note: sftNote},
	{runtime: sftLoRARuntime, name: ", LoRA adapter on conversations", note: sftLoRANote, lora: true},
}

// SFTWorkspace offers the Train tab a fine-tuning of the served model on a
// conversation dataset, whole or as a LoRA adapter: each conversation is
// rendered through the model's chat template, a few are held out, the model or
// its adapter trains on the rest, and the trained model and its held-out
// verdict are recorded as one run.
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

// WorkflowCapabilities lists the fine-tunings the Train tab offers: the served
// model on a conversation dataset, and a LoRA adapter over it.
func (workspace *SFTWorkspace) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || kind != WorkflowTraining {
		return nil, nil
	}
	var capabilities []WorkflowCapability
	for _, method := range sftMethods {
		program, err := ggufProgram(method.runtime, workspace.trainer.Model)
		if err != nil {
			return nil, err
		}
		controls := []workflowControl{
			{Name: "dataset", Type: workflowControlDataset, Required: true, Label: "conversations", Format: sftDatasetFormat},
			{Name: "steps", Type: workflowControlInteger, Required: true},
		}
		if method.lora {
			controls = append(controls, workflowControl{Name: "rank", Type: workflowControlInteger, Required: true, Bounds: &workflowControlBounds{Default: sftLoRADefaultRank, Step: 1}})
		}
		controls = append(controls,
			workflowControl{Name: "train_on_prompts", Type: workflowControlBoolean},
			workflowControl{Name: "average_weights", Type: workflowControlBoolean},
			workflowControl{Name: "sample_every", Type: workflowControlInteger},
			workflowControl{Name: "save_every", Type: workflowControlInteger},
			workflowControl{Name: "resume", Type: workflowControlOutput},
			workflowControl{Name: "output", Type: workflowControlOutput, Required: true})
		capabilities = append(capabilities, WorkflowCapability{
			Task: recipe.TaskTraining, Recipe: program.ID, Model: workspace.trainer.Model,
			Name: filepath.Base(workspace.trainer.File) + method.name, Location: workspace.trainer.File,
			Host:     trainsOnHost,
			Note:     method.note,
			Stages:   []recipe.Stage{{Node: recipe.Node{ID: "train", Module: recipe.ModuleID(method.runtime)}}},
			Outputs:  []recipe.Output{{Name: "model", Data: recipe.DataArtifact, Source: recipe.Endpoint{Node: "train", Port: "model"}}},
			Controls: controls,
		})
	}
	return capabilities, nil
}

// method is the fine-tuning the recipe names for the served model, with its program.
func (workspace *SFTWorkspace) method(recipeID artifact.ID) (sftMethod, artifact.Descriptor, error) {
	for _, method := range sftMethods {
		program, err := ggufProgram(method.runtime, workspace.trainer.Model)
		if err != nil {
			return sftMethod{}, artifact.Descriptor{}, err
		}
		if program.ID == recipeID {
			return method, program, nil
		}
	}
	return sftMethod{}, artifact.Descriptor{}, errors.New("sft workspace: workflow is not admitted")
}

type sftInput struct {
	Dataset artifact.ID `json:"dataset"`
	Steps   int         `json:"steps"`
	Rank    int         `json:"rank"`
	Output  string      `json:"output"`
	// TrainOnPrompts learns every token of each conversation rather than its replies.
	TrainOnPrompts bool `json:"train_on_prompts"`
	// AverageWeights publishes a moving average of the weights as the trained model.
	AverageWeights bool `json:"average_weights"`
	// SampleEvery, when positive, has the run generate after the prompts of held-out
	// conversations every that many updates, for the page to show beside the verdict.
	SampleEvery int `json:"sample_every"`
	// SaveEvery, when positive, saves a checkpoint after every that many updates, and Resume
	// names one, an output of an earlier run, to continue from.
	SaveEvery int    `json:"save_every"`
	Resume    string `json:"resume"`
}

// ExecuteWorkflow fine-tunes the served model on the dataset, whole or as an
// adapter of the rank given, and records the run: the model and the
// conversations in, the trained model and the verdict that judged it out. A run
// that fails or is cancelled records its outcome.
func (workspace *SFTWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, capability WorkflowCapability, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != WorkflowTraining || capability.Task != recipe.TaskTraining {
		return operation.Completion{}, errors.New("sft workspace: workflow is not admitted")
	}
	method, program, err := workspace.method(capability.Recipe)
	if err != nil {
		return operation.Completion{}, err
	}
	var input sftInput
	if err := strictjson.DecodeBytes(raw, &input); err != nil {
		return operation.Completion{}, err
	}
	inputs := []artifact.ID{workspace.trainer.Model, input.Dataset}
	if input.Dataset.Kind() != artifact.KindDataset || input.Steps <= 0 || input.SampleEvery < 0 || input.SaveEvery < 0 {
		return operation.Completion{}, errors.New("sft workspace: a conversation dataset and a positive number of steps are required, and samples and saved checkpoints come every positive number of steps")
	}
	var lora *trainingworkflow.LoRA
	switch {
	case method.lora && input.Rank <= 0:
		return operation.Completion{}, errors.New("sft workspace: a LoRA adapter needs a positive rank")
	case !method.lora && input.Rank != 0:
		return operation.Completion{}, errors.New("sft workspace: a rank belongs to the LoRA adapter, not to fine-tuning the whole model")
	case method.lora:
		lora = &trainingworkflow.LoRA{Rank: input.Rank, Alpha: float64(input.Rank * trainingworkflow.LoRAAlphaPerRank)}
	}
	output, err := outputDirectory(workspace.roots.Checkpoints, input.Output)
	var resume string
	if err == nil && strings.TrimSpace(input.Resume) != "" {
		resume, err = outputDirectory(workspace.roots.Checkpoints, input.Resume)
	}
	if err != nil {
		return failWorkflow(ctx, workspace.store, program.ID, inputs, "training_failed", err)
	}
	total := uint64(input.Steps)
	reporter.Progress(0, &total)
	result, err := trainingworkflow.RunConversationSFT(ctx, workspace.store, trainingworkflow.ConversationSFT{
		Dataset: input.Dataset, Render: workspace.trainer.Render, ModelDirectory: workspace.trainer.File,
		OutputDirectory: output, WorkDirectory: filepath.Join(workspace.roots.Datasets, "sft"),
		Steps: input.Steps, LoRA: lora, Prompts: input.TrainOnPrompts, Average: input.AverageWeights, SampleEvery: input.SampleEvery,
		CheckpointEvery: input.SaveEvery, CheckpointKeep: sftCheckpointsKept, Resume: resume,
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
	if result.Samples.Valid() {
		outputs = append(outputs, result.Samples)
	}
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
// run. Both ways of fine-tuning read the conversations alike.
func (workspace *SFTWorkspace) PreviewWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, position, limit int) (any, error) {
	if workspace == nil || kind != WorkflowTraining || task != recipe.TaskTraining {
		return nil, errors.New("sft workspace: workflow is not admitted")
	}
	if _, _, err := workspace.method(recipeID); err != nil {
		return nil, err
	}
	var input struct {
		Dataset        artifact.ID `json:"dataset"`
		TrainOnPrompts bool        `json:"train_on_prompts"`
	}
	if err := json.Unmarshal(raw, &input); err != nil || input.Dataset.Kind() != artifact.KindDataset {
		return nil, errors.Join(err, errors.New("sft workspace: preview needs a conversation dataset"))
	}
	return trainingworkflow.PreviewConversations(ctx, workspace.store, input.Dataset, workspace.trainer.Render, workspace.trainer.File, input.TrainOnPrompts, position, limit)
}

// outputDirectory is the directory a run named name writes under root: the
// name is one safe path component.
func outputDirectory(root, name string) (string, error) {
	name = strings.TrimSpace(name)
	if fsatomic.PathComponent(name) != nil {
		return "", errors.New("training workspace: output must be one safe path component")
	}
	return filepath.Join(root, name), nil
}
