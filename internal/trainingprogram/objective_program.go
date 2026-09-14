package trainingprogram

import "overgo/internal/hostoptimizer"

const (
	ObjectiveOperatorForward  = "objective-forward"
	ObjectiveOperatorBackward = "objective-backward"
	ObjectiveOperatorMuon     = "muon"
)

// CompileObjectiveProgram seals the shared forward/backward/Muon order.
func CompileObjectiveProgram(objective ObjectiveKind, parameters []ParameterSpec, plan hostoptimizer.Plan) (TrainingProgram, error) {
	if parameters == nil {
		parameters = make([]ParameterSpec, plan.GroupCount())
		for index := range parameters {
			group, _ := plan.Group(index)
			parameters[index] = ParameterSpec{
				Name: group.Name, Rows: group.Rows, Cols: group.Cols, Trainable: !group.Frozen,
			}
		}
	}
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
