package trainingprogram

import (
	"overgo/internal/hostoptimizer"
	"testing"
	"time"
)

func TestTrainingSessionPlanDerivesUpdatesAndSealsPolicy(t *testing.T) {
	const (
		datasetUnits    = 3
		maximumSequence = 128
	)
	config := hostoptimizer.Config{BaseLearningRate: 0.01, Momentum: 0.9, Schedule: hostoptimizer.ScheduleConstant}
	plan, err := CompileTrainingSessionPlan(SessionSpec{
		Objective: ObjectiveTokenPrediction, DatasetUnits: datasetUnits,
		MaximumSequence: maximumSequence, Optimizer: config,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Updates() != datasetUnits || plan.MaximumSequence() != maximumSequence ||
		plan.Optimizer().Steps != datasetUnits || !plan.ID().Valid() {
		t.Fatalf("plan = %#v", plan)
	}
	first, err := plan.Seed("init")
	if err != nil {
		t.Fatal(err)
	}
	second, err := plan.Seed("evaluation")
	if err != nil || first == second {
		t.Fatalf("stream seeds = %d/%d, err=%v", first, second, err)
	}
}

func TestTrainingSessionPlanValidatesRunControls(t *testing.T) {
	base := SessionSpec{
		Objective: ObjectiveGRPO, DatasetUnits: 1, MaximumSequence: 128, ObjectiveScale: 1,
		Optimizer: hostoptimizer.Config{BaseLearningRate: 0.01, Momentum: 0.9, Schedule: hostoptimizer.ScheduleConstant},
	}
	cases := []struct {
		name   string
		mutate func(*SessionSpec)
	}{
		{name: "negative updates", mutate: func(spec *SessionSpec) { spec.RequestedUpdates = -1 }},
		{name: "missing RL scale", mutate: func(spec *SessionSpec) { spec.ObjectiveScale = 0 }},
		{name: "negative wall", mutate: func(spec *SessionSpec) { spec.MaxProjectedWall = -time.Second }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			spec := base
			test.mutate(&spec)
			if _, err := CompileTrainingSessionPlan(spec); err == nil {
				t.Fatal("invalid session accepted")
			}
		})
	}
}
