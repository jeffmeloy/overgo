//go:build windows

package optimizer

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/devicemath"
	"overgo/internal/hostoptimizer"
)

// TestDeviceMuonMatrixStepMatchesHost gates one device (fp32) Muon matrix update
// against the host optimizer's Step (fp64 Newton-Schulz) for square/tall/wide
// groups: both the updated weights and the momentum must match within tolerance.
func TestDeviceMuonMatrixStepMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var ops *deviceOps
	if err := worker.Do(context.Background(), func(state *device.State) error { var err error; ops, err = newDeviceOps(state); return err }); err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(_ *device.State) error { ops.close(); return nil })

	const mu, baseLR = 0.9, 0.1
	cases := []struct {
		name       string
		rows, cols int
	}{
		{"square", 32, 32},
		{"tall", 64, 16},
		{"wide", 16, 64},
	}
	rng := rand.New(rand.NewSource(2))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := tc.rows * tc.cols
			w := make([]float32, n)
			g := make([]float32, n)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
				g[i] = float32(rng.NormFloat64())
			}

			// Host reference: one Muon matrix group, one Step.
			hostW := make([]float32, n)
			copy(hostW, w)
			hostG := make([]float32, n)
			copy(hostG, g)
			plan, err := CompilePlan(n, []GroupSpec{{Name: "w", Start: 0, End: n, Rows: tc.rows, Cols: tc.cols}})
			if err != nil {
				t.Fatal(err)
			}
			opt, err := New(hostW, hostG, plan, Config{BaseLearningRate: baseLR, Momentum: mu, Schedule: ScheduleConstant})
			if err != nil {
				t.Fatal(err)
			}
			opt.Step()

			// Device: same inputs, zero initial momentum.
			devW := make([]float32, n)
			copy(devW, w)
			devG := make([]float32, n)
			copy(devG, g)
			devM := make([]float32, n)
			if err := deviceMuonMatrixStep(worker, ops, devW, devG, devM, tc.rows, tc.cols, mu, baseLR, false); err != nil {
				t.Fatal(err)
			}

			var maxW, maxM float64
			for i := range n {
				if d := math.Abs(float64(devW[i]) - float64(hostW[i])); d > maxW {
					maxW = d
				}
				if d := math.Abs(float64(devM[i]) - opt.Momentum()[i]); d > maxM {
					maxM = d
				}
			}
			t.Logf("%s (%dx%d): max abs weight %.3e momentum %.3e", tc.name, tc.rows, tc.cols, maxW, maxM)
			// Observed weight ~2.4e-7 (fp32 NS precision), momentum exact.
			const tolerance = 1e-5
			if maxW > tolerance || maxM > tolerance {
				t.Fatalf("%s: weight %.3e / momentum %.3e > %.1e", tc.name, maxW, maxM, tolerance)
			}
		})
	}
	for _, tf32 := range []bool{false, true} {
		schedule := "8+2-FP32"
		if tf32 {
			schedule = "4+2-TF32"
		}
		for _, magnitude := range []float32{0, 1e-14} {
			t.Run(fmt.Sprintf("zero-weights/%s/gradient=%g", schedule, magnitude), func(t *testing.T) {
				const rows, cols = 8, 8
				g, m := make([]float32, rows*cols), make([]float32, rows*cols)
				for i := range g {
					g[i] = magnitude
					m[i] = magnitude / 2
				}
				gotW, gotG, gotM := make([]float32, len(g)), slices.Clone(g), slices.Clone(m)
				if err := deviceMuonMatrixStep(worker, ops, gotW, gotG, gotM, rows, cols, mu, baseLR, tf32); err != nil {
					t.Fatal(err)
				}
				// The guarded update retains the direction and still updates momentum and weights.
				weightScale := float32(-math.Sqrt(float64(max(rows, cols))) * hostoptimizer.StepRMS(mu) * baseLR)
				for i := range g {
					wantM := float32(float32(mu)*m[i]) + g[i]
					direction := float32(float32(mu)*wantM) + g[i]
					wantW := float32(weightScale*direction) + float32(0)
					if math.Float32bits(gotW[i]) != math.Float32bits(wantW) || math.Float32bits(gotM[i]) != math.Float32bits(wantM) || math.Float32bits(gotG[i]) != math.Float32bits(g[i]) {
						t.Fatalf("guarded update[%d] weights=%g want=%g momentum=%g want=%g gradient=%g want=%g", i, gotW[i], wantW, gotM[i], wantM, gotG[i], g[i])
					}
				}
			})
		}
	}

}

func TestDeviceMuonPlanResidentMatchesShared(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	const rows, cols, vector = 9, 7, 11
	matrix := rows * cols
	count := matrix + vector
	rng := rand.New(rand.NewSource(67))
	weights, gradients := make([]float32, count), make([]float32, count)
	for index := range count {
		weights[index], gradients[index] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	plan, err := CompilePlan(count, []GroupSpec{
		{Name: "matrix", Start: 0, End: matrix, Rows: rows, Cols: cols},
		{Name: "vector", Start: matrix, End: count, Rows: vector, Cols: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	config := Config{BaseLearningRate: 0.07, Momentum: 0.91, Steps: 3, Schedule: ScheduleLinearDecay}
	wantWeights, wantGradients, wantMomentum := append([]float32(nil), weights...), append([]float32(nil), gradients...), make([]float32, count)
	if err := DeviceMuonStepPlan(worker, wantWeights, wantGradients, wantMomentum, plan, 2, config); err != nil {
		t.Fatal(err)
	}
	weightsPtr, err := devicemath.AllocResidentF32(worker, count, weights)
	if err != nil {
		t.Fatal(err)
	}
	gradientsPtr, err := devicemath.AllocResidentF32(worker, count, gradients)
	if err != nil {
		_ = devicemath.FreeResident(worker, weightsPtr)
		t.Fatal(err)
	}
	momentumPtr, err := devicemath.AllocResidentF32(worker, count, nil)
	if err != nil {
		_ = devicemath.FreeResident(worker, weightsPtr, gradientsPtr)
		t.Fatal(err)
	}
	defer devicemath.FreeResident(worker, weightsPtr, gradientsPtr, momentumPtr)
	if err := DeviceMuonPlanResident(worker, weightsPtr, gradientsPtr, momentumPtr, plan, 2, config); err != nil {
		t.Fatal(err)
	}
	gotWeights, gotGradients, gotMomentum := make([]float32, count), make([]float32, count), make([]float32, count)
	for _, item := range []struct {
		pointer driver.DevicePtr
		data    []float32
	}{{weightsPtr, gotWeights}, {gradientsPtr, gotGradients}, {momentumPtr, gotMomentum}} {
		if err := devicemath.ReadResident(worker, item.pointer, devicemath.ResidentSlice{Data: item.data}); err != nil {
			t.Fatal(err)
		}
	}
	weightDelta, momentumDelta := maxF32Delta(gotWeights, wantWeights), maxF32Delta(gotMomentum, wantMomentum)
	gradientDelta := maxF32Delta(gotGradients, wantGradients)
	t.Logf("resident/shared Muon plan weights=%.3e momentum=%.3e gradient=%.3e", weightDelta, momentumDelta, gradientDelta)
	if weightDelta != 0 || momentumDelta != 0 || gradientDelta != 0 {
		t.Fatalf("resident Muon plan differs")
	}
}

func maxF32Delta(left, right []float32) float64 {
	var result float64
	for index := range left {
		result = max(result, math.Abs(float64(left[index]-right[index])))
	}
	return result
}

// TestDeviceMuonStepPlanMatchesHost gates matrix and vector Muon groups.
func TestDeviceMuonStepPlanMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const mu, baseLR = 0.9, 0.1
	const mR, mC, vN = 16, 16, 16
	matrixN := mR * mC
	n := matrixN + vN
	rng := rand.New(rand.NewSource(3))
	w := make([]float32, n)
	g := make([]float32, n)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
		g[i] = float32(rng.NormFloat64())
	}
	plan, err := CompilePlan(n, []GroupSpec{
		{Name: "m", Start: 0, End: matrixN, Rows: mR, Cols: mC},
		{Name: "v", Start: matrixN, End: n, Rows: vN, Cols: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	config := Config{BaseLearningRate: baseLR, Momentum: mu, Schedule: ScheduleConstant}

	hostW := make([]float32, n)
	copy(hostW, w)
	hostG := make([]float32, n)
	copy(hostG, g)
	opt, err := New(hostW, hostG, plan, config)
	if err != nil {
		t.Fatal(err)
	}
	opt.Step()

	devW := make([]float32, n)
	copy(devW, w)
	devG := make([]float32, n)
	copy(devG, g)
	devM := make([]float32, n)
	if err := DeviceMuonStepPlan(worker, devW, devG, devM, plan, 1, config); err != nil {
		t.Fatal(err)
	}

	var maxW, maxM float64
	for i := range n {
		if d := math.Abs(float64(devW[i]) - float64(hostW[i])); d > maxW {
			maxW = d
		}
		if d := math.Abs(float64(devM[i]) - opt.Momentum()[i]); d > maxM {
			maxM = d
		}
	}
	t.Logf("mixed plan: max abs weight %.3e momentum %.3e", maxW, maxM)
	const tolerance = 1e-5
	if maxW > tolerance || maxM > tolerance {
		t.Fatalf("weight %.3e / momentum %.3e > %.1e", maxW, maxM, tolerance)
	}
}

// DeviceMuonPlanResident applies and clears one flat resident Muon plan.
func DeviceMuonPlanResident(
	worker *device.Worker,
	weights, gradients, momentum driver.DevicePtr,
	plan Plan,
	step int,
	config Config,
) error {
	if worker == nil || weights == 0 || gradients == 0 || momentum == 0 || plan.Identity() == "" || step <= 0 {
		return fmt.Errorf("DeviceMuonPlanResident: invalid buffer, plan, or step")
	}
	if err := config.Validate(); err != nil {
		return err
	}
	session, err := NewResidentMuonPlan(worker, plan, config)
	if err != nil {
		return err
	}
	defer session.Close()
	return session.Step(weights, gradients, momentum, step)
}

// deviceMuonMatrixStep: staged single-matrix Muon update.
func deviceMuonMatrixStep(worker *device.Worker, ops *deviceOps, weights, gradient, momentum []float32, rows, cols int, mu, rate float64, tf32 bool) error {
	return worker.Do(context.Background(), func(_ *device.State) error {
		return ops.muonMatrixGroup(weights, gradient, momentum, rows, cols, mu, rate, tf32)
	})
}

// muonMatrixGroup: stage, update, return weights and momentum.
func (o *deviceOps) muonMatrixGroup(weights, gradient, momentum []float32, rows, cols int, mu, rate float64, tf32 bool) error {
	n, valid := hostoptimizer.MatrixElements(rows, cols)
	if !valid || len(weights) != n || len(gradient) != n || len(momentum) != n {
		return fmt.Errorf("muonMatrixGroup: shape mismatch (n=%d w=%d g=%d m=%d)", n, len(weights), len(gradient), len(momentum))
	}
	buffers, err := o.uploadF32Set(weights, gradient, momentum)
	if err != nil {
		return err
	}
	defer freeDevicePointers(o.lib, buffers...)
	dW, dG, dM := buffers[0], buffers[1], buffers[2]

	scratch, err := o.allocMuonScratch(n, min(rows, cols)*min(rows, cols))
	if err != nil {
		return err
	}
	defer scratch.free(o.lib)
	update := o.muonMatrixGroupResidentWithScratch
	if tf32 {
		update = o.muonMatrixGroupResidentTrainingWithScratch
	}
	if err := update(dW, dG, dM, rows, cols, mu, rate, scratch); err != nil {
		return err
	}
	if err := o.lib.StreamSynchronize(o.stream); err != nil {
		return err
	}
	if err := o.lib.MemcpyDtoH(driver.Bytes(weights), dW); err != nil {
		return err
	}
	return o.lib.MemcpyDtoH(driver.Bytes(momentum), dM)
}
