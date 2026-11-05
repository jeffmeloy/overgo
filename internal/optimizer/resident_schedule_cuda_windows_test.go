//go:build windows

package optimizer

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostoptimizer"
	"overgo/internal/testcheck"
)

// TestResidentScheduleIsRecordedAndJudged: the training plan names the schedule it steps by and the host's plan none, and
// a matrix of 1/i spectrum, where the schedules part most, holds the schedule to what it states: the device's 4+2 at TF32
// agrees with its host oracle, whose direction is shorter than the 8+2 direction by the factor the update's size takes.
func TestResidentScheduleIsRecordedAndJudged(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	testcheck.NoError(t, err)
	defer worker.Close()
	var ops *deviceOps
	if err := worker.Do(context.Background(), func(state *device.State) error {
		var err error
		ops, err = newDeviceOps(state)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(_ *device.State) error { ops.close(); return nil })

	const rows, cols = 64, 32
	matrices := []ResidentMatrix{{Weights: 1, Gradient: 2, Momentum: 3, Rows: rows, Cols: cols}}
	config := Config{BaseLearningRate: 0.1, Momentum: 0.9, Schedule: ScheduleConstant}
	for name, build := range map[string]struct {
		open func(*device.Worker, []ResidentMatrix, Config) (*ResidentMuonPlan, error)
		want string
	}{
		"training": {NewResidentMatrixMuonTrainingPlan, residentSchedule},
		"host":     {NewResidentMatrixMuonPlan, ""},
	} {
		plan, err := build.open(worker, matrices, config)
		testcheck.NoError(t, err)
		if got := plan.Schedule(); got != build.want {
			t.Errorf("the %s plan names schedule %q, want %q", name, got, build.want)
		}
		testcheck.NoError(t, plan.Close())
	}

	// A diagonal of 1/i turned by one reflection, so the spectrum is 1/i and no element stands alone.
	rng := rand.New(rand.NewSource(7))
	reflection := make([]float64, rows)
	var length float64
	for index := range reflection {
		reflection[index] = rng.NormFloat64()
		length += reflection[index] * reflection[index]
	}
	input := make([]float32, rows*cols)
	for column := range cols {
		singular := 1 / float64(column+1)
		for row := range rows {
			value := -2 * reflection[row] * reflection[column] * singular / length
			if row == column {
				value += singular
			}
			input[row*cols+column] = float32(value)
		}
	}
	var buffers nsBuffers
	if err := worker.Do(context.Background(), func(_ *device.State) error {
		var err error
		buffers, err = ops.allocBuffers(rows*cols, gramDim(rows, cols)*gramDim(rows, cols))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(_ *device.State) error { buffers.free(ops.lib); return nil })
	got, _, err := deviceNewtonSchulz(worker, ops, buffers, input, rows, cols, true)
	testcheck.NoError(t, err)
	short, full := make([]float64, len(input)), make([]float64, len(input))
	for index, value := range input {
		short[index], full[index] = float64(value), float64(value)
	}
	hostoptimizer.NewtonSchulz(short, rows, cols, residentNewtonSchulzStage1Iterations)
	hostoptimizer.NewtonSchulz(full, rows, cols, hostoptimizer.NewtonSchulzStage1Iterations)
	var difference, shortSquared, fullSquared float64
	for index := range short {
		difference += math.Pow(float64(got[index])-short[index], 2)
		shortSquared += short[index] * short[index]
		fullSquared += full[index] * full[index]
	}
	const oracleBound = 0.02
	if relative := math.Sqrt(difference / shortSquared); relative > oracleBound {
		t.Fatalf("the device's 4+2 stands %.4g from its host oracle, past %g", relative, oracleBound)
	}
	ratio := math.Sqrt(shortSquared / fullSquared)
	t.Logf("update size at 4+2 is %.4f of 8+2 on a 1/i spectrum of rank %d", ratio, cols)
	if ratio >= 1 {
		t.Fatalf("the short schedule's direction is %.4f of the full one's, want it shorter", ratio)
	}
}
