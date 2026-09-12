package hybridtrain

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/optimizer"
)

type segmentTrainer func(*Model, int, optimizer.Config, TrainingOptions) ([]float64, error)

func progressModel(t *testing.T) *Model {
	t.Helper()
	m, err := BuildModel(smallHybrid(), testSeed)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func verifyOptimizerResume(t *testing.T, train segmentTrainer) {
	t.Helper()
	for _, schedule := range []optimizer.Schedule{optimizer.ScheduleConstant, optimizer.ScheduleLinearDecay} {
		cfg := optimizer.Config{BaseLearningRate: testLearningRate, Momentum: testMomentum, Schedule: schedule, Steps: testSteps}
		whole := progressModel(t)
		var want optimizer.State
		losses, err := train(whole, testSteps, cfg, TrainingOptions{Checkpoint: func(s optimizer.State) error { want = s; return nil }})
		if err != nil {
			t.Fatal(err)
		}
		for _, split := range []int{0, 1, 3, testSteps - 1} {
			first := progressModel(t)
			var saved optimizer.State
			prefix, err := train(first, split, cfg, TrainingOptions{Checkpoint: func(s optimizer.State) error { saved = s; return nil }})
			if err != nil {
				t.Fatal(err)
			}
			// Restore into independently allocated masters and round-trip the existing
			// portable optimizer representation, not a live optimizer instance.
			restarted := progressModel(t)
			copy(restarted.matW, first.matW)
			copy(restarted.vecW, first.vecW)
			raw, err := json.Marshal(saved)
			if err != nil {
				t.Fatal(err)
			}
			var restored optimizer.State
			if err := json.Unmarshal(raw, &restored); err != nil {
				t.Fatal(err)
			}
			var got optimizer.State
			observed := split
			suffix, err := train(restarted, testSteps-split, cfg, TrainingOptions{
				Resume: &restored,
				Observe: func(step int, _ float64) error {
					if step != observed {
						t.Fatalf("observer step=%d want=%d", step, observed)
					}
					observed++
					return nil
				},
				Checkpoint: func(s optimizer.State) error { got = s; return nil },
			})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(append(prefix, suffix...), losses) || !slices.Equal(whole.matW, restarted.matW) || !slices.Equal(whole.vecW, restarted.vecW) || !reflect.DeepEqual(got, want) || observed != testSteps {
				t.Fatalf("schedule=%v split=%d: loss, masters, momentum, schedule or observation progress differs", schedule, split)
			}
			if !reflect.DeepEqual(saved, restored) {
				t.Fatal("resume modified caller-owned momentum")
			}
		}
	}
}

func TestHybridOptimizerResumeHost(t *testing.T) {
	verifyOptimizerResume(t, (*Model).TrainHost)
}

func verifyOptimizerStop(t *testing.T, train segmentTrainer) {
	t.Helper()
	cfg := optimizer.Config{BaseLearningRate: testLearningRate, Momentum: testMomentum}
	m := progressModel(t)
	stop, publication := errors.New("operator stop"), errors.New("checkpoint publication")
	var saved optimizer.State
	called := 0
	losses, err := train(m, testSteps, cfg, TrainingOptions{
		Observe:    func(int, float64) error { return stop },
		Checkpoint: func(s optimizer.State) error { saved = s; called++; return publication },
	})
	if !errors.Is(err, stop) || !errors.Is(err, publication) || len(losses) != 1 || saved.Step != 1 || called != 1 {
		t.Fatalf("stop boundary: loss=%v state=%d callbacks=%d error=%v", losses, saved.Step, called, err)
	}
	whole := progressModel(t)
	want, err := train(whole, testSteps, cfg, TrainingOptions{})
	if err != nil {
		t.Fatal(err)
	}
	more, err := train(m, testSteps-1, cfg, TrainingOptions{Resume: &saved})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(append(losses, more...), want) || !slices.Equal(m.matW, whole.matW) || !slices.Equal(m.vecW, whole.vecW) {
		t.Fatal("observer stop did not expose matching weight/optimizer state")
	}
}

func TestHybridOptimizerStopHost(t *testing.T) { verifyOptimizerStop(t, (*Model).TrainHost) }

func TestHybridResumeRefusals(t *testing.T) {
	m := progressModel(t)
	cfg := optimizer.Config{BaseLearningRate: testLearningRate, Momentum: testMomentum}
	var baseline optimizer.State
	if _, err := m.TrainHost(0, cfg, TrainingOptions{Checkpoint: func(s optimizer.State) error { baseline = s; return nil }}); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*optimizer.State){
		"plan":          func(s *optimizer.State) { s.PlanIdentity = "invalid" },
		"momentum-size": func(s *optimizer.State) { s.Momentum = s.Momentum[:1] },
		"negative-step": func(s *optimizer.State) { s.Step = -1 },
		"overflow":      func(s *optimizer.State) { s.Step = math.MaxInt },
		"config":        func(s *optimizer.State) { s.Config.Momentum = 0 },
		"nan":           func(s *optimizer.State) { s.Momentum[0] = math.NaN() },
		"infinite":      func(s *optimizer.State) { s.Momentum[0] = math.Inf(1) },
	} {
		t.Run(name, func(t *testing.T) {
			state := baseline
			state.Momentum = slices.Clone(state.Momentum)
			mutate(&state)
			verifyResumeRefused(t, m, 1, cfg, TrainingOptions{Resume: &state})
		})
	}
	verifyResumeRefused(t, m, -1, cfg, TrainingOptions{})
	badConfig := cfg
	badConfig.Momentum = math.NaN()
	verifyResumeRefused(t, m, 1, badConfig, TrainingOptions{})
	baseline.Momentum[0] = 1.0 / 3.0
	for _, train := range deviceSegmentTrainers(nil) {
		if _, err := train(m, 1, cfg, TrainingOptions{Resume: &baseline}); err == nil {
			t.Fatal("lossy f32 resume accepted")
		}
	}
}

func verifyResumeRefused(t *testing.T, m *Model, steps int, cfg optimizer.Config, options TrainingOptions) {
	t.Helper()
	beforeMat, beforeVec := slices.Clone(m.matW), slices.Clone(m.vecW)
	called := false
	options.Checkpoint = func(optimizer.State) error { called = true; return nil }
	for _, train := range append(deviceSegmentTrainers(nil), (*Model).TrainHost) {
		if _, err := train(m, steps, cfg, options); err == nil {
			t.Fatal("invalid progress accepted")
		}
	}
	if called || !slices.Equal(beforeMat, m.matW) || !slices.Equal(beforeVec, m.vecW) {
		t.Fatal("refused resume mutated or published state")
	}
}

type failedBoundary struct {
	phase   string
	failure error
	steps   []int
}

func (b *failedBoundary) Forward() ([]float32, error) {
	if b.phase == "forward" {
		return nil, b.failure
	}
	return []float32{1}, nil
}
func (b *failedBoundary) Backward([]float32) error {
	if b.phase == "backward" {
		return b.failure
	}
	return nil
}
func (b *failedBoundary) Step(step int) error {
	b.steps = append(b.steps, step)
	if b.phase == "update" {
		return b.failure
	}
	return nil
}

func TestHybridTrainingBoundaryFailures(t *testing.T) {
	m := progressModel(t)
	failure := errors.New("incomplete update")
	for _, phase := range []string{"forward", "backward", "update"} {
		backend := &failedBoundary{phase: phase, failure: failure}
		observed := false
		trajectory, err := runTrainingObserved(m.program, backend, []float32{0}, 2, 3, func(int, float64) error { observed = true; return nil })
		if !errors.Is(err, failure) || trajectory != nil || observed {
			t.Fatalf("%s exposed an incomplete update boundary: %v %v", phase, trajectory, err)
		}
		if phase == "update" && !slices.Equal(backend.steps, []int{4}) {
			t.Fatal("resumed update index restarted")
		}
	}
	backend := &failedBoundary{}
	trajectory, err := runTrainingObserved(m.program, backend, []float32{0}, 2, 3, func(step int, _ float64) error {
		if step != 3 {
			t.Fatalf("absolute observer index=%d", step)
		}
		return failure
	})
	if !errors.Is(err, failure) || len(trajectory) != 1 || !slices.Equal(backend.steps, []int{4}) {
		t.Fatal("complete update boundary was discarded")
	}
}
