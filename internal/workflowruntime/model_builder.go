package workflowruntime

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

// ModelBuildState: durable phase identities.
type ModelBuildState struct {
	Recipe       artifact.ID `json:"recipe"`
	Dataset      artifact.ID `json:"dataset"`
	Construction artifact.ID `json:"construction"`
	Model        artifact.ID `json:"model"`
	Checkpoint   artifact.ID `json:"checkpoint,omitempty"`
	Run          artifact.ID `json:"run,omitempty"`
	Evaluation   artifact.ID `json:"evaluation,omitempty"`
	Evidence     artifact.ID `json:"evidence,omitempty"`
	Decision     artifact.ID `json:"decision,omitempty"`
}

// ModelBuildSession: one recipe-selected construction campaign.
type ModelBuildSession interface {
	Initialize(context.Context) (ModelBuildState, error)
	Train(context.Context, ModelBuildState) (ModelBuildState, error)
	Evaluate(context.Context, ModelBuildState) (ModelBuildState, error)
	Record(context.Context, ModelBuildState) (ModelBuildState, error)
	Promote(context.Context, ModelBuildState) (ModelBuildState, error)
}

func ExecuteModelBuild(ctx context.Context, session ModelBuildSession) (ModelBuildState, error) {
	if ctx == nil || session == nil {
		return ModelBuildState{}, errors.New("model builder: context and session required")
	}
	state, err := session.Initialize(ctx)
	if err != nil {
		return ModelBuildState{}, err
	}
	if err = validateBuildState(state, 0); err != nil {
		return ModelBuildState{}, err
	}
	for phase, execute := range []func(context.Context, ModelBuildState) (ModelBuildState, error){
		session.Train, session.Evaluate, session.Record, session.Promote,
	} {
		next, executeErr := execute(ctx, state)
		if executeErr != nil {
			return state, executeErr
		}
		if next.Recipe != state.Recipe || next.Dataset != state.Dataset || next.Construction != state.Construction ||
			validateBuildState(next, phase+1) != nil {
			return state, errors.New("model builder: phase changed authority or omitted output")
		}
		state = next
	}
	return state, nil
}

func validateBuildState(state ModelBuildState, phase int) error {
	valid := state.Recipe.Kind() == artifact.KindRecipe && state.Dataset.Kind() == artifact.KindDataset &&
		state.Construction.Kind() == artifact.KindRecipe && state.Model.Kind() == artifact.KindModel
	if phase >= 1 {
		valid = valid && state.Checkpoint.Kind() == artifact.KindCheckpoint
	}
	if phase >= 2 {
		valid = valid && state.Run.Kind() == artifact.KindRun && state.Evaluation.Kind() == artifact.KindEvaluation
	}
	if phase >= 3 {
		valid = valid && state.Evidence.Kind() == artifact.KindEvidence
	}
	if phase >= 4 {
		valid = valid && state.Decision.Kind() == artifact.KindEvidence
	}
	if !valid {
		return errors.New("model builder: incomplete phase state")
	}
	return nil
}
