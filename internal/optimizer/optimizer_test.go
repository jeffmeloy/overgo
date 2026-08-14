package optimizer

import (
	"math"
	"slices"
	"testing"
)

const (
	fixtureBaseLearningRate = 0.01
	fixtureMomentum         = 0.95
	fixtureSteps            = 10
	fixtureTolerance        = 1e-7
)

var fixtureConfig = Config{
	BaseLearningRate: fixtureBaseLearningRate,
	Momentum:         fixtureMomentum,
	Steps:            fixtureSteps,
	Schedule:         ScheduleLinearDecay,
}

func TestCompilePlanCompilesAllGeometryAndIdentity(t *testing.T) {
	const (
		matrixRows = 2
		matrixCols = 3
		vectorCols = 2
	)
	matrixElements := matrixRows * matrixCols
	parameterCount := matrixElements + vectorCols
	specs := []GroupSpec{
		{Name: "matrix", Start: 0, End: matrixElements, Rows: matrixRows, Cols: matrixCols},
		{Name: "vector", Start: matrixElements, End: parameterCount, Rows: 1, Cols: vectorCols},
	}
	plan, err := CompilePlan(parameterCount, specs)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ParameterCount() != parameterCount || plan.GroupCount() != len(specs) {
		t.Fatalf("plan dimensions = (%d,%d), want (%d,%d)", plan.ParameterCount(), plan.GroupCount(), parameterCount, len(specs))
	}
	matrix, _ := plan.Group(0)
	vector, _ := plan.Group(1)
	if matrix.Rows != matrixRows || matrix.Cols != matrixCols || vector.Rows != 1 || vector.Cols != vectorCols {
		t.Fatalf("compiled geometry = %dx%d / %dx%d", matrix.Rows, matrix.Cols, vector.Rows, vector.Cols)
	}
	duplicate, err := CompilePlan(parameterCount, specs)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Identity() == "" || plan.Identity() != duplicate.Identity() {
		t.Fatal("equivalent plans lack a stable identity")
	}
	specs[1].Frozen = true
	frozen, err := CompilePlan(parameterCount, specs)
	if err != nil {
		t.Fatal(err)
	}
	if frozen.Identity() == plan.Identity() {
		t.Fatal("frozen policy is absent from plan identity")
	}
}

func TestCompilePlanRejectsInvalidLayouts(t *testing.T) {
	const parameterCount = 4
	cases := map[string][]GroupSpec{
		"unnamed":    {{Start: 0, End: parameterCount, Rows: 2, Cols: 2}},
		"gap":        {{Name: "matrix", Start: 1, End: parameterCount, Rows: 1, Cols: 3}},
		"shape":      {{Name: "matrix", Start: 0, End: parameterCount, Rows: 1, Cols: 3}},
		"incomplete": {{Name: "matrix", Start: 0, End: 2, Rows: 1, Cols: 2}},
		"duplicate": {
			{Name: "same", Start: 0, End: 2, Rows: 1, Cols: 2},
			{Name: "same", Start: 2, End: parameterCount, Rows: 1, Cols: 2},
		},
	}
	for name, specs := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := CompilePlan(parameterCount, specs); err == nil {
				t.Fatal("invalid plan accepted")
			}
		})
	}
}

func TestScheduleUsesExplicitStepSemantics(t *testing.T) {
	if got, want := fixtureConfig.LearningRate(1), fixtureBaseLearningRate*(fixtureSteps-1)/fixtureSteps; math.Abs(got-want) > fixtureTolerance {
		t.Fatalf("first-step rate = %g, want %g", got, want)
	}
	if got := fixtureConfig.LearningRate(fixtureSteps); got != 0 {
		t.Fatalf("terminal rate = %g, want 0", got)
	}
	constant := fixtureConfig
	constant.Schedule = ScheduleConstant
	if got := constant.LearningRate(fixtureSteps + 1); got != fixtureBaseLearningRate {
		t.Fatalf("constant rate = %g, want %g", got, fixtureBaseLearningRate)
	}
}

func TestNewRejectsUncompiledPlan(t *testing.T) {
	if _, err := New(nil, nil, Plan{}, Config{}); err == nil {
		t.Fatal("uncompiled plan accepted")
	}
}

func TestVectorMuonStepCarriesMomentum(t *testing.T) {
	const parameterCount = 2
	plan := mustPlan(t, parameterCount, []GroupSpec{{Name: "norm", Start: 0, End: parameterCount, Rows: 1, Cols: parameterCount}})
	weights := []float32{1, -1}
	gradients := []float32{1, -1}
	config := fixtureConfig
	config.Schedule = ScheduleConstant
	optimizer := mustOptimizer(t, weights, gradients, plan, config)
	optimizer.Step()
	want := slices.Clone(weights)
	if want[0] >= 1 || want[1] <= -1 {
		t.Fatalf("vector Muon did not follow gradient: %v", want)
	}
	optimizer.Step()
	if slices.Equal(weights, want) {
		t.Fatalf("vector Muon did not carry momentum: %v", weights)
	}
	if gradients != nil && !slices.Equal(gradients, make([]float32, parameterCount)) {
		t.Fatalf("vector Muon retained gradients: %v", gradients)
	}
}

func TestMuonStepIsScaleInvariantAndConsumesGradients(t *testing.T) {
	const (
		rows           = 2
		cols           = 3
		parameterCount = rows * cols
		gradientScale  = 100
	)
	plan := mustPlan(t, parameterCount, []GroupSpec{{Name: "attention", Start: 0, End: parameterCount, Rows: rows, Cols: cols}})
	baseGradient := []float32{1, -2, 3, -4, 5, -6}
	scaledGradient := make([]float32, parameterCount)
	for index, value := range baseGradient {
		scaledGradient[index] = gradientScale * value
	}
	baseWeights := make([]float32, parameterCount)
	scaledWeights := make([]float32, parameterCount)
	base := mustOptimizer(t, baseWeights, baseGradient, plan, fixtureConfig)
	scaled := mustOptimizer(t, scaledWeights, scaledGradient, plan, fixtureConfig)
	baseResult := base.Step()
	scaledResult := scaled.Step()
	if baseResult != scaledResult || baseResult.Step != 1 {
		t.Fatalf("step results differ: base=%+v scaled=%+v", baseResult, scaledResult)
	}
	for index := range parameterCount {
		if difference := math.Abs(float64(baseWeights[index] - scaledWeights[index])); difference > fixtureTolerance {
			t.Fatalf("scale-invariant weight[%d] difference = %g", index, difference)
		}
		if baseGradient[index] != 0 || scaledGradient[index] != 0 {
			t.Fatalf("gradient[%d] was not consumed", index)
		}
	}
}

func TestFrozenGroupDoesNotMoveAndConsumesGradients(t *testing.T) {
	const parameterCount = 4
	plan := mustPlan(t, parameterCount, []GroupSpec{{
		Name: "frozen", Start: 0, End: parameterCount, Rows: 2, Cols: 2, Frozen: true,
	}})
	weights := []float32{1, 2, 3, 4}
	want := slices.Clone(weights)
	gradients := []float32{4, 3, 2, 1}
	optimizer := mustOptimizer(t, weights, gradients, plan, fixtureConfig)
	optimizer.Step()
	if !slices.Equal(weights, want) {
		t.Fatalf("frozen weights moved: got %v want %v", weights, want)
	}
	if !slices.Equal(gradients, make([]float32, parameterCount)) {
		t.Fatalf("frozen gradients retained: %v", gradients)
	}
}

func TestStateRestoresExactContinuation(t *testing.T) {
	const parameterCount = 4
	plan := mustPlan(t, parameterCount, []GroupSpec{{Name: "matrix", Start: 0, End: parameterCount, Rows: 2, Cols: 2}})
	firstWeights := []float32{1, 2, 3, 4}
	firstGradients := []float32{1, -1, 2, -2}
	first := mustOptimizer(t, firstWeights, firstGradients, plan, fixtureConfig)
	first.Step()

	secondWeights := slices.Clone(firstWeights)
	secondGradients := make([]float32, parameterCount)
	second := mustOptimizer(t, secondWeights, secondGradients, plan, fixtureConfig)
	if err := second.Restore(first.Snapshot()); err != nil {
		t.Fatal(err)
	}
	nextGradient := []float32{-2, 3, -4, 5}
	copy(firstGradients, nextGradient)
	copy(secondGradients, nextGradient)
	if first.Step() != second.Step() {
		t.Fatal("restored step result differs")
	}
	if !slices.Equal(firstWeights, secondWeights) {
		t.Fatalf("restored continuation differs: first=%v second=%v", firstWeights, secondWeights)
	}

	state := first.Snapshot()
	state.Config.BaseLearningRate *= 2
	if err := second.Restore(state); err == nil {
		t.Fatal("mismatched config accepted")
	}
}

func mustPlan(t *testing.T, parameterCount int, specs []GroupSpec) Plan {
	t.Helper()
	plan, err := CompilePlan(parameterCount, specs)
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func mustOptimizer(t *testing.T, weights, gradients []float32, plan Plan, config Config) *Optimizer {
	t.Helper()
	optimizer, err := New(weights, gradients, plan, config)
	if err != nil {
		t.Fatal(err)
	}
	return optimizer
}
