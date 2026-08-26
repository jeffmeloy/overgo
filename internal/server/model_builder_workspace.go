package server

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelbuilder"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/scratchmodel"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

type ModelBuilderWorkspace struct {
	store  artifact.Repository
	recipe artifact.ID
}

func NewModelBuilderWorkspace(ctx context.Context, store artifact.Repository) (*ModelBuilderWorkspace, error) {
	if store == nil {
		return nil, errors.New("model builder workspace: repository required")
	}
	profile, err := scratchmodel.ResolveActiveDerivationProfile(ctx, store)
	if err != nil {
		return nil, err
	}
	recipeID, err := modelbuilder.ScratchRecipeID(profile.ID)
	if err != nil {
		return nil, err
	}
	return &ModelBuilderWorkspace{store: store, recipe: recipeID}, nil
}

func (workspace *ModelBuilderWorkspace) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || kind != WorkflowModelBuild {
		return nil, nil
	}
	return []WorkflowCapability{{
		Task: recipe.TaskTraining, Recipe: workspace.recipe,
		Stages: workflowruntime.ModelBuildStages(),
		Outputs: []recipe.Output{
			{Name: "model", Data: recipe.DataArtifact, Source: recipe.Endpoint{Node: "construct", Port: "model"}},
			{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "construct", Port: "checkpoint"}},
		},
		Controls: []WorkflowControl{
			{Name: "corpus", Type: WorkflowControlText, Required: true},
			{Name: "seed", Type: WorkflowControlInteger, Required: true},
			{Name: "steps", Type: WorkflowControlInteger, Required: true},
		},
	}}, nil
}

type modelBuildInput struct {
	Corpus string `json:"corpus"`
	Seed   int64  `json:"seed"`
	Steps  int    `json:"steps"`
}

func (workspace *ModelBuilderWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != WorkflowModelBuild ||
		task != recipe.TaskTraining || recipeID != workspace.recipe {
		return operation.Completion{}, errors.New("model builder workspace: workflow is not admitted")
	}
	var input modelBuildInput
	if err := strictjson.DecodeBytes(raw, &input); err != nil {
		return operation.Completion{}, err
	}
	documents := nonemptyLines(input.Corpus)
	session, err := modelbuilder.NewScratchSession(ctx, modelbuilder.ScratchRequest{
		Repository: workspace.store, Documents: documents, Seed: input.Seed, Steps: input.Steps,
	})
	if err != nil {
		return operation.Completion{}, err
	}
	state, err := workflowruntime.ExecuteModelBuild(ctx, workspace.store, reporter.OperationID(), session)
	if err != nil {
		return operation.Completion{}, err
	}
	reporter.Publishing()
	total := uint64(1)
	reporter.Progress(total, &total)
	return operation.Completion{Run: state.Run, Outputs: []artifact.ID{
		state.Model, state.Checkpoint, state.Evaluation, state.Evidence, state.Decision,
	}}, nil
}

func nonemptyLines(text string) []string {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	documents := lines[:0]
	for _, line := range lines {
		if line = strings.TrimSpace(line); line != "" {
			documents = append(documents, line)
		}
	}
	return documents
}
