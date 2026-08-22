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

type modelBuildPhase struct {
	execute  func(context.Context, ModelBuildState) (ModelBuildState, error)
	complete func(ModelBuildState) bool
}

func ExecuteModelBuild(ctx context.Context, session ModelBuildSession) (ModelBuildState, error) {
	if ctx == nil || session == nil {
		return ModelBuildState{}, errors.New("model builder: context and session required")
	}
	state, err := session.Initialize(ctx)
	if err != nil {
		return ModelBuildState{}, err
	}
	if err = validateBuildState(state); err != nil {
		return ModelBuildState{}, err
	}
	var required []func(ModelBuildState) bool
	for _, phase := range modelBuildPhases(session) {
		next, executeErr := phase.execute(ctx, state)
		if executeErr != nil {
			return state, executeErr
		}
		required = append(required, phase.complete)
		if next.Recipe != state.Recipe || next.Dataset != state.Dataset || next.Construction != state.Construction ||
			validateBuildState(next, required...) != nil {
			return state, errors.New("model builder: phase changed authority or omitted output")
		}
		state = next
	}
	return state, nil
}

func modelBuildPhases(session ModelBuildSession) []modelBuildPhase {
	return []modelBuildPhase{
		{execute: session.Train, complete: func(state ModelBuildState) bool {
			return state.Checkpoint.Kind() == artifact.KindCheckpoint
		}},
		{execute: session.Evaluate, complete: func(state ModelBuildState) bool {
			return state.Run.Kind() == artifact.KindRun && state.Evaluation.Kind() == artifact.KindEvaluation
		}},
		{execute: session.Record, complete: func(state ModelBuildState) bool {
			return state.Evidence.Kind() == artifact.KindEvidence
		}},
		{execute: session.Promote, complete: func(state ModelBuildState) bool {
			return state.Decision.Kind() == artifact.KindEvidence
		}},
	}
}

func validateBuildState(state ModelBuildState, required ...func(ModelBuildState) bool) error {
	valid := state.Recipe.Kind() == artifact.KindRecipe && state.Dataset.Kind() == artifact.KindDataset &&
		state.Construction.Kind() == artifact.KindRecipe && state.Model.Kind() == artifact.KindModel
	for _, complete := range required {
		valid = valid && complete(state)
	}
	if !valid {
		return errors.New("model builder: incomplete phase state")
	}
	return nil
}
