package trainingprogram

import (
	"slices"
	"testing"

	"overgo/internal/optimizer"
)

func TestObjectiveExecutionRunsCompiledOrderWithoutAllocation(t *testing.T) {
	plan, err := optimizer.CompilePlan(1, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileObjectiveProgram(ObjectiveForecast, nil, plan)
	if err != nil {
		t.Fatal(err)
	}
	var order []OperatorPhase
	execution, err := BindObjective(
		program,
		func(*struct{}) error { order = append(order, PhaseForward); return nil },
		func(*struct{}) error { order = append(order, PhaseBackward); return nil },
		func(*struct{}) error { order = append(order, PhaseOptimize); return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	state := struct{}{}
	order = make([]OperatorPhase, 0, 3)
	if allocations := testing.AllocsPerRun(100, func() {
		order = order[:0]
		if err := execution.Run(&state); err != nil {
			panic(err)
		}
	}); allocations != 0 {
		t.Fatalf("allocations per complete execution = %g", allocations)
	}
	if want := []OperatorPhase{PhaseForward, PhaseBackward, PhaseOptimize}; !slices.Equal(order, want) {
		t.Fatalf("execution order = %v, want %v", order, want)
	}
}
