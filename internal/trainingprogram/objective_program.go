package trainingprogram

import "overgo/internal/optimizer"

const (
	ObjectiveOperatorForward  = "objective-forward"
	ObjectiveOperatorBackward = "objective-backward"
	ObjectiveOperatorMuon     = "muon"
)

// CompileObjectiveProgram seals the shared forward/backward/Muon order.
func CompileObjectiveProgram(objective ObjectiveKind, parameters []ParameterSpec, plan optimizer.Plan) (TrainingProgram, error) {
	return CompileTrainingProgram(ProgramSpec{
		Objective: objective,
		Operators: []OperatorSpec{
			{ID: ObjectiveOperatorForward, Phase: PhaseForward},
			{ID: ObjectiveOperatorBackward, Phase: PhaseBackward},
			{ID: ObjectiveOperatorMuon, Phase: PhaseOptimize},
		},
		Parameters: parameters,
		Optimizer:  plan,
	})
}
