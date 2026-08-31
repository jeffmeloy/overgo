package oscillatorimage

import (
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/optimizer"
	"overgo/internal/testevidence"
)

func testTrainingConfig() optimizer.Config {
	return optimizer.Config{BaseLearningRate: 0.02, Momentum: 0.95, Schedule: optimizer.ScheduleConstant}
}

func TestTrainerDecreasesBootstrapLossTiny(t *testing.T) {
	m := tinyModel()
	trainer, err := NewTrainer(m, testTrainingConfig(), 11)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	before := trainer.Loss(7)
	var first, last TrainStepResult
	for step := range 8 {
		result, err := trainer.Step()
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			t.Fatalf("step %d loss non-finite: %+v", step, result)
		}
		if step == 0 {
			first = result
			if result.GradientL2 <= 0 {
				t.Fatalf("first step carried no gradient: %+v", result)
			}
		}
		last = result
	}
	after := trainer.Loss(7)
	if !(after < before) {
		t.Fatalf("bootstrap loss did not descend: %.6f -> %.6f (steps %.6f -> %.6f)", before, after, first.Loss, last.Loss)
	}
	if first.Step != 1 || last.Step != 8 {
		t.Fatalf("step facts wrong: first=%+v last=%+v", first, last)
	}
}

func TestTrainerUpdatesModelFieldsInPlace(t *testing.T) {
	m := tinyModel()
	trainer, err := NewTrainer(m, testTrainingConfig(), 11)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	before := slices.Clone(m.Omega)
	total := 0
	for _, tensor := range m.trainableLayout() {
		total += len(*tensor.field)
	}
	if total != trainer.ParameterCount() {
		t.Fatalf("layout covers %d of %d packed parameters", total, trainer.ParameterCount())
	}
	if _, err := trainer.Step(); err != nil {
		t.Fatal(err)
	}
	if slices.Equal(before, m.Omega) {
		t.Fatal("optimizer update did not reach model fields")
	}
}

func TestTrainerRefusesGradientsOutsidePack(t *testing.T) {
	m := tinyModel()
	trainer, err := NewTrainer(m, testTrainingConfig(), 11)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	trainer.grads[m.tensorName("stray")] = make([]float32, 4)
	if _, err := trainer.Step(); err == nil || !strings.Contains(err.Error(), "outside the trainable pack") {
		t.Fatalf("stray gradient slot was not refused: %v", err)
	}
}

// TestRealArtifactBootstrapTraining verifies objective band and scratch descent.
func TestRealArtifactBootstrapTraining(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	real, err := Load(artifactDir(t))
	if err != nil {
		t.Skipf("UNAVAILABLE: Un-0 artifact absent: %v", err)
	}
	realTrainer, err := NewTrainer(real, testTrainingConfig(), 202)
	if err != nil {
		t.Fatal(err)
	}
	defer realTrainer.Close()
	shipped := realTrainer.Loss(202)
	if shipped < 15.57-3*1.16 || shipped > 15.57+3*1.16 {
		t.Fatalf("real checkpoint loss %.4f outside the reference shipped band 15.57 +/- 3*1.16; objective identity not established", shipped)
	}
	t.Logf("objective identity: real checkpoint loss %.4f inside shipped band; parameters=%d", shipped, realTrainer.ParameterCount())

	const steps = 12
	scratch := NewBootstrapModel(real.Cfg, 17)
	trainer, err := NewTrainer(scratch, testTrainingConfig(), 17)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	before := trainer.Loss(202)
	start := time.Now()
	for step := range steps {
		result, err := trainer.Step()
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) || (step == 0 && result.GradientL2 <= 0) {
			t.Fatalf("step %d invalid: %+v", step+1, result)
		}
		t.Logf("step %d/%d: loss=%.6f lr=%.4g grad_l2=%.4g", step+1, steps, result.Loss, result.LearningRate, result.GradientL2)
	}
	after := trainer.Loss(202)
	if math.IsNaN(after) || math.IsInf(after, 0) || !(after < before) {
		t.Fatalf("scratch bootstrap loss did not descend: %.6f -> %.6f", before, after)
	}
	t.Logf("scratch bootstrap training: steps=%d loss %.6f -> %.6f total_wall=%s",
		steps, before, after, time.Since(start).Round(time.Millisecond))
}
