package trainingprogram

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

// Binding supplies one compiled operator implementation.
type Binding[State any] struct {
	Operator string
	Execute  func(*State) error
}

type boundOperator[State any] struct {
	spec    OperatorSpec
	execute func(*State) error
}

// Execution is a fully bound TrainingProgram.
type Execution[State any] struct {
	programID artifact.ID
	operators []boundOperator[State]
}

// BindObjective binds standard forward/backward/Muon operators.
func BindObjective[State any](
	program TrainingProgram,
	forward, backward, optimize func(*State) error,
) (Execution[State], error) {
	return Bind(program, []Binding[State]{
		{Operator: ObjectiveOperatorForward, Execute: forward},
		{Operator: ObjectiveOperatorBackward, Execute: backward},
		{Operator: ObjectiveOperatorMuon, Execute: optimize},
	})
}

// Bind requires one implementation for every compiled operator.
func Bind[State any](program TrainingProgram, bindings []Binding[State]) (Execution[State], error) {
	if program.ID().Kind() != artifact.KindRecipe || len(program.operators) == 0 {
		return Execution[State]{}, errors.New("training program: compiled authority absent")
	}
	byID := make(map[string]func(*State) error, len(bindings))
	for _, binding := range bindings {
		if binding.Operator == "" || binding.Execute == nil {
			return Execution[State]{}, errors.New("training program: invalid operator binding")
		}
		if _, duplicate := byID[binding.Operator]; duplicate {
			return Execution[State]{}, fmt.Errorf("training program: duplicate binding %q", binding.Operator)
		}
		byID[binding.Operator] = binding.Execute
	}
	operators := make([]boundOperator[State], len(program.operators))
	for index, spec := range program.operators {
		execute, ok := byID[spec.ID]
		if !ok {
			return Execution[State]{}, fmt.Errorf("training program: binding %q absent", spec.ID)
		}
		operators[index] = boundOperator[State]{spec: spec, execute: execute}
		delete(byID, spec.ID)
	}
	for id := range byID {
		return Execution[State]{}, fmt.Errorf("training program: binding %q not declared", id)
	}
	return Execution[State]{programID: program.ID(), operators: operators}, nil
}

func (e Execution[State]) ProgramID() artifact.ID { return e.programID }

// Run executes the complete compiled sequence.
func (e Execution[State]) Run(state *State) error {
	if state == nil || e.programID.Kind() != artifact.KindRecipe || len(e.operators) == 0 {
		return errors.New("training program: bound execution unavailable")
	}
	for _, operator := range e.operators {
		if err := operator.execute(state); err != nil {
			return fmt.Errorf("training program: %s: %w", operator.spec.ID, err)
		}
	}
	return nil
}

// Select compiles a phase subset once.
func (e Execution[State]) Select(phases ...OperatorPhase) (Execution[State], error) {
	if e.programID.Kind() != artifact.KindRecipe || len(e.operators) == 0 || len(phases) == 0 {
		return Execution[State]{}, errors.New("training program: bound execution unavailable")
	}
	selected := make(map[OperatorPhase]struct{}, len(phases))
	for _, phase := range phases {
		if !validPhase(phase) {
			return Execution[State]{}, fmt.Errorf("training program: invalid execution phase %q", phase)
		}
		selected[phase] = struct{}{}
	}
	operators := make([]boundOperator[State], 0, len(e.operators))
	for _, operator := range e.operators {
		if _, ok := selected[operator.spec.Phase]; ok {
			operators = append(operators, operator)
		}
	}
	if len(operators) == 0 {
		return Execution[State]{}, errors.New("training program: selected execution is empty")
	}
	return Execution[State]{programID: e.programID, operators: operators}, nil
}
