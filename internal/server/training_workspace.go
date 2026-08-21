package server

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
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
)

type TrainingWorkspace struct {
	store     artifact.Repository
	roots     dataroot.Roots
	program   recipe.Program
	objective trainingprogram.ObjectiveKind
}

func NewTrainingWorkspace(ctx context.Context, store artifact.Repository, roots dataroot.Roots, model artifact.ID) (*TrainingWorkspace, error) {
	if ctx == nil || store == nil || model.Kind() != artifact.KindModel || roots.Checkpoints == "" {
		return nil, errors.New("training workspace: invalid store, roots, or model")
	}
	_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, model, recipe.TaskTraining)
	if err != nil {
		return nil, err
	}
	objective, err := trainingworkflow.ProgramObjective(program)
	if err != nil {
		return nil, err
	}
	if objective != trainingprogram.ObjectiveDPO && objective != trainingprogram.ObjectiveGRPO {
		return nil, errors.New("training workspace: active recipe is not an RL objective")
	}
	return &TrainingWorkspace{store: store, roots: roots, program: program, objective: objective}, nil
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
			{Name: "objective_scale", Type: WorkflowControlNumber, Required: true},
		},
	}}, nil
}

type trainingWorkflowInput struct {
	Dataset        artifact.ID `json:"dataset"`
	Resume         artifact.ID `json:"resume,omitempty"`
	Output         string      `json:"output"`
	Steps          int         `json:"steps"`
	ObjectiveScale float64     `json:"objective_scale"`
}

func (workspace *TrainingWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != WorkflowTraining || task != recipe.TaskTraining || recipeID != workspace.program.Definition().ID {
		return operation.Completion{}, errors.New("training workspace: workflow is not admitted")
	}
	var input trainingWorkflowInput
	if err := strictjson.DecodeBytes(raw, &input); err != nil {
		return operation.Completion{}, err
	}
	definition := workspace.program.Definition()
	policy, ok := definition.Dependency(recipe.DependencyModel, 0)
	if !ok {
		return operation.Completion{}, errors.New("training workspace: policy dependency absent")
	}
	policyDirectory, err := artifact.AvailablePath(ctx, workspace.store, policy, artifact.LocationDirectory)
	if err != nil {
		return workspace.fail(ctx, definition.ID, []artifact.ID{policy, input.Dataset}, err)
	}
	var reference artifact.ID
	referenceDirectory := ""
	if workspace.objective == trainingprogram.ObjectiveDPO {
		reference, ok = definition.Dependency(recipe.DependencyModel, 1)
		if !ok {
			return operation.Completion{}, errors.New("training workspace: reference dependency absent")
		}
		referenceDirectory, err = artifact.AvailablePath(ctx, workspace.store, reference, artifact.LocationDirectory)
		if err != nil {
			return workspace.fail(ctx, definition.ID, []artifact.ID{policy, reference, input.Dataset}, err)
		}
	}
	if input.Dataset.Kind() != artifact.KindDataset {
		return operation.Completion{}, errors.New("training workspace: dataset identity required")
	}
	datasetPath, err := artifact.AvailablePath(ctx, workspace.store, input.Dataset, artifact.LocationFile)
	if err != nil {
		return workspace.fail(ctx, definition.ID, []artifact.ID{policy, input.Dataset}, err)
	}
	resumeDirectory := ""
	inputs := []artifact.ID{policy}
	if reference.Valid() {
		inputs = append(inputs, reference)
	}
	inputs = append(inputs, input.Dataset)
	if input.Resume.Valid() {
		resumeDirectory, err = artifact.AvailablePath(ctx, workspace.store, input.Resume, artifact.LocationDirectory)
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
	var completed uint64
	dpo := make([]trainingprogram.DPOObservation, 0, input.Steps)
	grpo := make([]trainingprogram.GRPOObservation, 0, input.Steps)
	result, executeErr := trainingworkflow.Execute(ctx, trainingworkflow.Request{
		Repository:     workspace.store,
		Recipe:         recipeID,
		ModelDirectory: policyDirectory, ReferenceDirectory: referenceDirectory,
		DatasetPath: datasetPath, OutputDirectory: output, ResumeDirectory: resumeDirectory,
		Steps:          input.Steps,
		ObjectiveScale: input.ObjectiveScale, Host: true,
		ObserveDPO: func(observation trainingprogram.DPOObservation) {
			dpo = append(dpo, observation)
			completed++
			reporter.Progress(completed, &total)
			reportDPO(reporter, observation)
		},
		ObserveGRPO: func(observation trainingprogram.GRPOObservation) {
			grpo = append(grpo, observation)
			completed++
			reporter.Progress(completed, &total)
			reportGRPO(reporter, observation)
		},
	})
	if executeErr != nil {
		return workspace.fail(ctx, definition.ID, inputs, executeErr)
	}
	reporter.Publishing()
	return workspace.publish(ctx, definition.ID, inputs, policy, reference, input.Dataset,
		result.Checkpoint, output, dpo, grpo)
}

func (workspace *TrainingWorkspace) outputPath(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" || name == "." || name == ".." || filepath.Base(name) != name || strings.ContainsAny(name, "\x00\r\n") {
		return "", errors.New("training workspace: output must be one safe path component")
	}
	return filepath.Join(workspace.roots.Checkpoints, name), nil
}

func (workspace *TrainingWorkspace) publish(ctx context.Context, recipeID artifact.ID, inputs []artifact.ID, policy, reference, dataset artifact.ID, checkpoint trainingprogram.Checkpoint, path string, dpo []trainingprogram.DPOObservation, grpo []trainingprogram.GRPOObservation) (operation.Completion, error) {
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
	evaluators := make([]artifact.ID, 0)
	seen := make(map[artifact.ID]struct{})
	for _, observation := range grpo {
		if _, ok := seen[observation.Evaluator]; !ok {
			seen[observation.Evaluator] = struct{}{}
			evaluators = append(evaluators, observation.Evaluator)
		}
	}
	trace, err := runrecord.NewTrainingTrace(run.ID, recipeID, dataset, policy, reference, evaluators, dpo, grpo)
	if err != nil {
		return operation.Completion{}, err
	}
	traceContent, err := trace.Content()
	if err != nil {
		return operation.Completion{}, err
	}
	decision, err := runrecord.NewTrainingDecision(run.ID, recipeID, policy, checkpoint.ID(), trace.ID)
	if err != nil {
		return operation.Completion{}, err
	}
	decisionContent, err := decision.Content()
	if err != nil {
		return operation.Completion{}, err
	}
	batch.Contents = append(batch.Contents, traceContent, decisionContent)
	batch.Lineage = append(batch.Lineage, trace.Lineage()...)
	batch.Lineage = append(batch.Lineage, decision.Lineage()...)
	batch.Locations = append(batch.Locations, artifact.LocationEvent{
		Location: artifact.Location{Artifact: checkpoint.ID(), Kind: artifact.LocationDirectory, Value: path},
		Action:   artifact.LocationAdd,
	})
	if _, err := artifact.CommitBatch(ctx, workspace.store, batch); err != nil {
		return operation.Completion{}, err
	}
	return operation.Completion{Run: run.ID, Outputs: []artifact.ID{checkpoint.ID(), trace.ID, decision.ID}}, nil
}

func reportGRPO(reporter operation.Reporter, observation trainingprogram.GRPOObservation) {
	for _, metric := range []operation.Metric{
		{Name: "grpo_loss", Value: observation.Loss},
		{Name: "mean_reward", Value: observation.MeanReward},
		{Name: "reward_dispersion", Value: observation.RewardDispersion},
		{Name: "group_size", Value: float64(observation.GroupSize), Unit: "rollouts"},
		{Name: "completion_tokens", Value: float64(observation.CompletionTokens), Unit: "tokens"},
		{Name: "learning_rate", Value: observation.LearningRate},
		{Name: "gradient_l2", Value: observation.GradientL2},
		{Name: "update_l2", Value: observation.UpdateL2},
	} {
		reporter.Metric(metric)
	}
}

func reportDPO(reporter operation.Reporter, observation trainingprogram.DPOObservation) {
	for _, metric := range []operation.Metric{
		{Name: "dpo_loss", Value: observation.Loss},
		{Name: "policy_margin", Value: observation.PolicyMargin},
		{Name: "reference_margin", Value: observation.ReferenceMargin},
		{Name: "relative_margin", Value: observation.RelativeMargin},
		{Name: "learning_rate", Value: observation.LearningRate},
		{Name: "gradient_l2", Value: observation.GradientL2},
		{Name: "update_l2", Value: observation.UpdateL2},
		{Name: "chosen_tokens", Value: float64(observation.ChosenTokens), Unit: "tokens"},
		{Name: "rejected_tokens", Value: float64(observation.RejectedTokens), Unit: "tokens"},
	} {
		reporter.Metric(metric)
	}
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
