package trainingprogram

import (
	"overgo/internal/hostoptimizer"
	"overgo/internal/testcheck"
	"slices"
	"testing"
)

func TestObjectiveExecutionRunsCompiledOrderWithoutAllocation(t *testing.T) {
	plan := testcheck.Must(hostoptimizer.CompilePlan(1, []hostoptimizer.GroupSpec{{Name: "weight", Start: 0, End: 1, Rows: 1, Cols: 1}}))(t)
	program := testcheck.Must(CompileObjectiveProgram(ObjectiveForecast, nil, plan))(t)
	var order []OperatorPhase
	execution := testcheck.Must(BindObjective(
		program,
		func(*struct{}) error { order = append(order, PhaseForward); return nil },
		func(*struct{}) error { order = append(order, PhaseBackward); return nil },
		func(*struct{}) error { order = append(order, PhaseOptimize); return nil },
	))(t)
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
