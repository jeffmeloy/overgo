//go:build windows

package scratchmodel

import (
	"math"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
)

func TestScratchResidentTrajectoryParity(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps})
	if err != nil {
		t.Fatal(err)
	}
	const steps = 3
	host, err := construction.TrainShared(steps)
	if err != nil {
		t.Fatal(err)
	}
	initializedAt := time.Now()
	trainer, err := NewResidentTrainer(construction, steps)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	initWall := time.Since(initializedAt)
	losses, walls := make([]float64, steps), make([]time.Duration, steps)
	for step := range steps {
		tokens, err := construction.Tokens(construction.split.Train[step%len(construction.split.Train)])
		if err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		losses[step], err = trainer.Step(tokens, step+1)
		walls[step] = time.Since(started)
		if err != nil {
			t.Fatal(err)
		}
	}
	validationTokens, err := construction.Tokens(construction.split.Validation[0])
	if err != nil {
		t.Fatal(err)
	}
	validation, err := trainer.Evaluate(validationTokens)
	if err != nil {
		t.Fatal(err)
	}
	worst := math.Abs(validation - host.ValidationLoss)
	for step := range steps {
		worst = max(worst, math.Abs(losses[step]-host.Losses[step]))
	}
	weights, momentum, err := trainer.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	for _, slab := range [][]float32{weights, momentum} {
		for _, value := range slab {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("resident scratch state is non-finite")
			}
		}
	}
	t.Logf("resident scratch init=%s cold=%s warm=%s/%s losses=%v validation=%.9f host max delta=%.3e", initWall, walls[0], walls[1], walls[2], losses, validation, worst)
	if worst > 3e-3 {
		t.Fatalf("resident scratch trajectory delta %.3e", worst)
	}
}
