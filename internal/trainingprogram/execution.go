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

// RunPhases executes selected phases in compiled order.
func (e Execution[State]) RunPhases(state *State, phases ...OperatorPhase) error {
	if state == nil || e.programID.Kind() != artifact.KindRecipe || len(e.operators) == 0 || len(phases) == 0 {
		return errors.New("training program: bound execution unavailable")
	}
	selected := make(map[OperatorPhase]struct{}, len(phases))
	for _, phase := range phases {
		if !validPhase(phase) {
			return fmt.Errorf("training program: invalid execution phase %q", phase)
		}
		selected[phase] = struct{}{}
	}
	for _, operator := range e.operators {
		if _, ok := selected[operator.spec.Phase]; !ok {
			continue
		}
		if err := operator.execute(state); err != nil {
			return fmt.Errorf("training program: %s: %w", operator.spec.ID, err)
		}
	}
	return nil
}
