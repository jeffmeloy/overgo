package workflowruntime

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/workflowrecipe"
)

const (
	modelBuildStateMediaType = "application/vnd.overgo.model-build-state+json"
	modelBuildStateSchema    = "overgo/model-build-state/v1"
	modelBuildStatePort      = recipe.PortName("state")
)

var modelBuildStateContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: modelBuildStateMediaType, Schema: modelBuildStateSchema,
}

// ModelBuildState contains durable construction outputs.
type ModelBuildState struct {
	Recipe       artifact.ID `json:"recipe"`
	Dataset      artifact.ID `json:"dataset"`
	Construction artifact.ID `json:"construction"`
	Model        artifact.ID `json:"model"`
	Checkpoint   artifact.ID `json:"checkpoint,omitzero"`
	Run          artifact.ID `json:"run,omitzero"`
	Evaluation   artifact.ID `json:"evaluation,omitzero"`
	Evidence     artifact.ID `json:"evidence,omitzero"`
	Decision     artifact.ID `json:"decision,omitzero"`
}

// ModelBuildSession binds mathematical stages to the shared recipe runtime.
type ModelBuildSession interface {
	Initialize(context.Context) (ModelBuildState, error)
	Train(context.Context, ModelBuildState) (ModelBuildState, error)
	Evaluate(context.Context, ModelBuildState) (ModelBuildState, error)
	Record(context.Context, ModelBuildState) (ModelBuildState, error)
	Promote(context.Context, ModelBuildState) (ModelBuildState, error)
}

type modelBuildStage struct {
	node    recipe.Node
	execute func(context.Context, ModelBuildState) (ModelBuildState, error)
	valid   func(ModelBuildState) bool
}

// ModelBuildStages returns the neutral compiled stage contracts.
func ModelBuildStages() []recipe.Stage {
	stages := modelBuildStageDefinitions(nil)
	result := make([]recipe.Stage, len(stages))
	for index, stage := range stages {
		module, _ := workflowrecipe.Module(stage.node.Module)
		result[index] = recipe.Stage{Node: stage.node, Module: module}
	}
	return result
}

// ExecuteModelBuild runs construction through the shared receipt runtime.
func ExecuteModelBuild(
	ctx context.Context,
	store artifact.Repository,
	operation artifact.ID,
	session ModelBuildSession,
) (ModelBuildState, error) {
	if ctx == nil || store == nil || operation.Kind() != artifact.KindEvidence || session == nil {
		return ModelBuildState{}, errors.New("model builder: context, repository, operation, and session required")
	}
	initial, err := session.Initialize(ctx)
	if err != nil || validateBuildState(initial) != nil {
		return ModelBuildState{}, errors.Join(err, validateBuildState(initial))
	}
	program, err := compileModelBuildProgram(initial)
	if err != nil {
		return ModelBuildState{}, err
	}
	runtime, err := NewForProgram(store, program)
	if err != nil {
		return ModelBuildState{}, err
	}
	for _, stage := range modelBuildStageDefinitions(session) {
		execute, valid := stage.execute, stage.valid
		if err := RegisterContextStage(runtime, stage.node.Module, initial.Model,
			func(ctx context.Context, state ModelBuildState) (ModelBuildState, error) {
				next, executeErr := execute(ctx, state)
				if executeErr != nil {
					return state, executeErr
				}
				if next.Recipe != state.Recipe || next.Dataset != state.Dataset || next.Construction != state.Construction || !valid(next) {
					return state, errors.New("model builder: stage changed authority or omitted output")
				}
				return next, nil
			}, func(state ModelBuildState) (artifact.Content, error) {
				return artifact.JSONContent(modelBuildStateContract, state)
			}); err != nil {
			return ModelBuildState{}, err
		}
	}
	content, err := artifact.JSONContent(modelBuildStateContract, initial)
	if err != nil {
		return ModelBuildState{}, err
	}
	result, err := runtime.ExecuteProgram(
		ctx, "model-build/"+operation.String(), operation, nil, program,
		map[recipe.PortName]Value{modelBuildStatePort: ArtifactValue(recipe.DataArtifact, initial, content)},
	)
	if err != nil {
		return ModelBuildState{}, err
	}
	datum, one := result.Outputs[modelBuildStatePort].Single()
	if !one {
		return ModelBuildState{}, errors.New("model builder: final state is not scalar")
	}
	if state, ok := datum.Value.(ModelBuildState); ok {
		return state, nil
	}
	var state ModelBuildState
	if datum.Content == nil || strictjson.DecodeBytes(datum.Content.Data, &state) != nil {
		return ModelBuildState{}, errors.New("model builder: final state is unavailable")
	}
	return state, nil
}

func compileModelBuildProgram(state ModelBuildState) (recipe.Program, error) {
	stages := modelBuildStageDefinitions(nil)
	nodes := make([]recipe.Node, len(stages))
	edges := make([]recipe.Edge, 0, len(stages)-1)
	for index, stage := range stages {
		nodes[index] = stage.node
		if index != 0 {
			edges = append(edges, recipe.Edge{
				From: recipe.Endpoint{Node: nodes[index-1].ID, Port: modelBuildStatePort},
				To:   recipe.Endpoint{Node: stage.node.ID, Port: modelBuildStatePort},
			})
		}
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: state.Model},
			{Role: recipe.DependencyDataset, Artifact: state.Dataset},
		}, nodes, edges,
		[]recipe.Input{{Name: modelBuildStatePort, Data: recipe.DataArtifact, Target: recipe.Endpoint{Node: nodes[0].ID, Port: modelBuildStatePort}}},
		[]recipe.Output{{Name: modelBuildStatePort, Data: recipe.DataArtifact, Source: recipe.Endpoint{Node: nodes[len(nodes)-1].ID, Port: modelBuildStatePort}}},
	)
	if err != nil {
		return recipe.Program{}, err
	}
	return recipe.CompileProgram(definition, workflowrecipe.Catalog())
}

func modelBuildStageDefinitions(session ModelBuildSession) []modelBuildStage {
	stage := func(id recipe.NodeID, module recipe.ModuleID) modelBuildStage {
		return modelBuildStage{node: recipe.Node{ID: id, Module: module, Placement: recipe.PlacementHost}}
	}
	stages := []modelBuildStage{
		stage("fit", workflowrecipe.ModuleFitModel),
		stage("evaluate", workflowrecipe.ModuleEvaluateModel),
		stage("record", workflowrecipe.ModuleRecordModel),
		stage("promote", workflowrecipe.ModulePromoteModel),
	}
	stages[0].valid = func(state ModelBuildState) bool {
		return state.Model.Kind() == artifact.KindModel && state.Checkpoint.Kind() == artifact.KindCheckpoint
	}
	stages[1].valid = func(state ModelBuildState) bool {
		return stages[0].valid(state) && state.Run.Kind() == artifact.KindRun && state.Evaluation.Kind() == artifact.KindEvaluation
	}
	stages[2].valid = func(state ModelBuildState) bool {
		return stages[1].valid(state) && state.Evidence.Kind() == artifact.KindEvidence
	}
	stages[3].valid = func(state ModelBuildState) bool {
		return stages[2].valid(state) && state.Decision.Kind() == artifact.KindEvidence
	}
	if session != nil {
		stages[0].execute = session.Train
		stages[1].execute = session.Evaluate
		stages[2].execute = session.Record
		stages[3].execute = session.Promote
	}
	return stages
}

func validateBuildState(state ModelBuildState) error {
	if state.Recipe.Kind() != artifact.KindRecipe || state.Dataset.Kind() != artifact.KindDataset ||
		state.Construction.Kind() != artifact.KindRecipe || state.Model.Kind() != artifact.KindModel {
		return errors.New("model builder: incomplete initial state")
	}
	return nil
}
