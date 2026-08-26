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
)

var modelBuildStateContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: modelBuildStateMediaType, Schema: modelBuildStateSchema,
}

// ModelBuildState contains durable construction outputs.
type ModelBuildState struct {
	Recipe            artifact.ID `json:"recipe"`
	Dataset           artifact.ID `json:"dataset"`
	DerivationProfile artifact.ID `json:"derivation_profile"`
	Optimizer         artifact.ID `json:"optimizer"`
	Construction      artifact.ID `json:"construction"`
	Model             artifact.ID `json:"model"`
	Checkpoint        artifact.ID `json:"checkpoint,omitzero"`
	Run               artifact.ID `json:"run,omitzero"`
	Evaluation        artifact.ID `json:"evaluation,omitzero"`
	Evidence          artifact.ID `json:"evidence,omitzero"`
	Decision          artifact.ID `json:"decision,omitzero"`
}

// ModelBuildSession binds mathematical stages to the shared recipe runtime.
type ModelBuildSession interface {
	Initialize(context.Context) (ModelBuildState, error)
	Train(context.Context, ModelBuildState) (ModelBuildState, error)
	Evaluate(context.Context, ModelBuildState) (ModelBuildState, error)
	Record(context.Context, ModelBuildState) (ModelBuildState, error)
	Promote(context.Context, ModelBuildState) (ModelBuildState, error)
}

// ModelBuildStages returns module-linked construction contracts.
func ModelBuildStages() []recipe.Stage {
	var stages []recipe.Stage
	seen := map[recipe.ModuleID]bool{}
	for id := workflowrecipe.ModuleFitModel; id != ""; {
		module, found := workflowrecipe.Module(id)
		if !found || module.StageNode == "" || seen[id] {
			panic("model builder: invalid stage module chain")
		}
		seen[id] = true
		stages = append(stages, recipe.Stage{
			Node: recipe.Node{ID: module.StageNode, Module: id, Placement: recipe.PlacementHost}, Module: module,
		})
		id = module.Next
	}
	return stages
}

func (state ModelBuildState) artifact(name recipe.PortName) (artifact.ID, bool) {
	switch name {
	case workflowrecipe.BuildRecipeFact:
		return state.Recipe, true
	case workflowrecipe.BuildDatasetFact:
		return state.Dataset, true
	case workflowrecipe.BuildOptimizerFact:
		return state.Optimizer, true
	case workflowrecipe.BuildConstructionFact:
		return state.Construction, true
	case workflowrecipe.BuildModelFact:
		return state.Model, true
	case workflowrecipe.BuildCheckpointFact:
		return state.Checkpoint, true
	case workflowrecipe.BuildRunFact:
		return state.Run, true
	case workflowrecipe.BuildEvaluationFact:
		return state.Evaluation, true
	case workflowrecipe.BuildEvidenceFact:
		return state.Evidence, true
	case workflowrecipe.BuildDecisionFact:
		return state.Decision, true
	default:
		return artifact.ID{}, false
	}
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
	if err != nil || validateBuildInputs(initial) != nil {
		return ModelBuildState{}, errors.Join(err, validateBuildInputs(initial))
	}
	program, err := workflowrecipe.Catalog().CompileLinear(
		recipe.TaskTraining,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: initial.Model},
			{Role: recipe.DependencyDataset, Artifact: initial.Dataset},
			{Role: recipe.DependencyDerivationProfile, Artifact: initial.DerivationProfile},
			{Role: recipe.DependencyOptimizer, Artifact: initial.Optimizer},
		},
		ModelBuildStages(),
	)
	if err != nil {
		return ModelBuildState{}, err
	}
	initial.Recipe = program.Definition().ID
	if err := validateBuildState(initial); err != nil {
		return ModelBuildState{}, err
	}
	runtime, err := NewForProgram(store, program)
	if err != nil {
		return ModelBuildState{}, err
	}
	executors := map[recipe.ModuleID]func(context.Context, ModelBuildState) (ModelBuildState, error){
		workflowrecipe.ModuleFitModel:      session.Train,
		workflowrecipe.ModuleEvaluateModel: session.Evaluate,
		workflowrecipe.ModuleRecordModel:   session.Record,
		workflowrecipe.ModulePromoteModel:  session.Promote,
	}
	for _, stage := range program.Stages() {
		execute, found := executors[stage.Module.ID]
		if !found {
			return ModelBuildState{}, errors.New("model builder: compiled stage has no executor")
		}
		module := stage.Module
		if err := RegisterContextStage(runtime, module.ID, initial.Model,
			func(ctx context.Context, state ModelBuildState) (ModelBuildState, error) {
				next, executeErr := execute(ctx, state)
				if executeErr != nil {
					return state, executeErr
				}
				if err := validateArtifactRequirements(module.Postconditions, state, next); err != nil {
					return state, err
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
		map[recipe.PortName]Value{workflowrecipe.ModelBuildStatePort: ArtifactValue(recipe.DataArtifact, initial, content)},
	)
	if err != nil {
		return ModelBuildState{}, err
	}
	datum, one := result.Outputs[workflowrecipe.ModelBuildStatePort].Single()
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

func validateArtifactRequirements(
	requirements []recipe.ArtifactRequirement,
	before, after ModelBuildState,
) error {
	for _, requirement := range requirements {
		got, found := after.artifact(requirement.Name)
		if !found || got.Kind() != requirement.Kind {
			return errors.New("model builder: stage omitted required artifact")
		}
		if requirement.Preserve {
			prior, found := before.artifact(requirement.Name)
			if !found || got != prior {
				return errors.New("model builder: stage changed authority")
			}
		}
	}
	return nil
}

func validateBuildState(state ModelBuildState) error {
	if state.Recipe.Kind() != artifact.KindRecipe || validateBuildInputs(state) != nil {
		return errors.New("model builder: incomplete initial state")
	}
	return nil
}

func validateBuildInputs(state ModelBuildState) error {
	if state.Dataset.Kind() != artifact.KindDataset || state.DerivationProfile.Kind() != artifact.KindProfile ||
		state.Optimizer.Kind() != artifact.KindProfile || state.Construction.Kind() != artifact.KindRecipe ||
		state.Model.Kind() != artifact.KindModel {
		return errors.New("model builder: incomplete initial state")
	}
	return nil
}
