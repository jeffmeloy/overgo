//go:build windows

package hybridtrain

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/optimizer"
)

// smallHybrid covers both mix operators.
func smallHybrid() StackConfig {
	return StackConfig{
		Types:  []LayerKind{FullAttention, LinearAttention},
		Tokens: 6, Hidden: 16, Inter: 24, Eps: 1e-6,
		Heads: 2, KVHeads: 1, HeadDim: 8, RopeDim: 4, RopeTheta: 10000,
		GDNKeyHeads: 2, GDNValueHeads: 2, GDNHeadDim: 4, GDNConvK: 3,
	}
}

// TestHybridTrainHostMasterStreamedMatchesHost verifies the host-master
// streamed lane reproduces the host trajectory (same 8+2 FP32 Newton-Schulz
// schedule, so parity is tight).
func TestHybridTrainHostMasterStreamedMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	cfg := smallHybrid()
	ocfg := optimizer.Config{BaseLearningRate: testLearningRate, Momentum: testMomentum, Schedule: optimizer.ScheduleConstant}

	host, err := BuildModel(cfg, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	streamed, err := BuildModel(cfg, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	hostTraj, err := host.TrainHost(testSteps, ocfg)
	if err != nil {
		t.Fatal(err)
	}
	observed := 0
	streamedTraj, err := streamed.TrainHostMasterStreamed(worker, testSteps, ocfg, func(step int, loss float64) error {
		if step != observed {
			t.Fatalf("observer saw step %d, want %d", step, observed)
		}
		observed++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if observed != testSteps {
		t.Fatalf("observer saw %d steps, want %d", observed, testSteps)
	}
	var maxRel float64
	for i := range hostTraj {
		r := math.Abs(hostTraj[i]-streamedTraj[i]) / (math.Abs(hostTraj[i]) + 1e-9)
		maxRel = max(maxRel, r)
		t.Logf("step %d  host=%.6f  streamed=%.6f  rel=%.3e", i, hostTraj[i], streamedTraj[i], r)
	}
	if !(streamedTraj[len(streamedTraj)-1] < streamedTraj[0]) {
		t.Fatalf("streamed loss did not decrease: %.6f -> %.6f", streamedTraj[0], streamedTraj[len(streamedTraj)-1])
	}
	const relTol = 5e-3
	if maxRel > relTol {
		t.Errorf("streamed trajectory parity maxRel %.3e > %.1e", maxRel, relTol)
	}
}

// TestHybridTrainLayerStreamedMatchesFullSlab proves the layer-streamed
// lifecycle (per-layer gradient scratch, Muon step fused into the backward
// walk) produces the SAME trajectory as the full-slab host-master lane. The
// per-group Muon math is identical work on identical inputs — a layer's step
// applies only after its own backward consumed its pre-step weights — so the
// tolerance is exact: bitwise-equal losses.
func TestHybridTrainLayerStreamedMatchesFullSlab(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	cfg := smallHybrid()
	ocfg := optimizer.Config{BaseLearningRate: testLearningRate, Momentum: testMomentum, Schedule: optimizer.ScheduleConstant}

	fullSlab, err := BuildModel(cfg, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	layered, err := BuildModel(cfg, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	fullTraj, err := fullSlab.TrainHostMasterStreamed(worker, testSteps, ocfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	observed := 0
	layerTraj, err := layered.TrainHostMasterLayerStreamed(worker, testSteps, ocfg, func(step int, loss float64) error {
		if step != observed {
			t.Fatalf("observer saw step %d, want %d", step, observed)
		}
		observed++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if observed != testSteps {
		t.Fatalf("observer saw %d steps, want %d", observed, testSteps)
	}
	for i := range fullTraj {
		t.Logf("step %d  full-slab=%.9f  layer-streamed=%.9f", i, fullTraj[i], layerTraj[i])
		if layerTraj[i] != fullTraj[i] {
			t.Errorf("step %d trajectory diverged: full-slab %.17g, layer-streamed %.17g", i, fullTraj[i], layerTraj[i])
		}
	}
	if !(layerTraj[len(layerTraj)-1] < layerTraj[0]) {
		t.Fatalf("layer-streamed loss did not decrease: %.6f -> %.6f", layerTraj[0], layerTraj[len(layerTraj)-1])
	}
	// The trained masters must agree too, not just the loss scalars.
	for i := range fullSlab.matW {
		if fullSlab.matW[i] != layered.matW[i] {
			t.Fatalf("trained matrix masters diverged at element %d: %g vs %g", i, fullSlab.matW[i], layered.matW[i])
		}
	}
	for i := range fullSlab.vecW {
		if fullSlab.vecW[i] != layered.vecW[i] {
			t.Fatalf("trained vector params diverged at element %d: %g vs %g", i, fullSlab.vecW[i], layered.vecW[i])
		}
	}
}

// TestHybridTrainDeviceResidentMatchesHost verifies trajectory and residency parity.
func TestHybridTrainDeviceResidentMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	cfg := smallHybrid()
	ocfg := optimizer.Config{BaseLearningRate: testLearningRate, Momentum: testMomentum, Schedule: optimizer.ScheduleConstant}

	host, err := BuildModel(cfg, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := BuildModel(cfg, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("resident matrix params=%d  host-Sign vector params=%d", host.MatrixParamCount(), host.VectorParamCount())

	hostTraj, err := host.TrainHost(testSteps, ocfg)
	if err != nil {
		t.Fatal(err)
	}
	devTraj, acc, err := dev.TrainDeviceResident(worker, testSteps, ocfg)
	if err != nil {
		t.Fatal(err)
	}

	// Trajectory parity.
	var maxAbs, maxRel float64
	for i := range hostTraj {
		d := math.Abs(hostTraj[i] - devTraj[i])
		r := d / (math.Abs(hostTraj[i]) + 1e-9)
		maxAbs = max(maxAbs, d)
		maxRel = max(maxRel, r)
		t.Logf("step %d  host=%.6f  device=%.6f  |diff|=%.3e  rel=%.3e", i, hostTraj[i], devTraj[i], d, r)
	}
	t.Logf("trajectory parity over %d steps: max|diff|=%.3e  maxRel=%.3e", testSteps, maxAbs, maxRel)

	// Descent.
	if !(hostTraj[len(hostTraj)-1] < hostTraj[0]) {
		t.Fatalf("host loss did not decrease: %.6f -> %.6f", hostTraj[0], hostTraj[len(hostTraj)-1])
	}
	if !(devTraj[len(devTraj)-1] < devTraj[0]) {
		t.Fatalf("device loss did not decrease: %.6f -> %.6f", devTraj[0], devTraj[len(devTraj)-1])
	}

	// Relative trajectory parity.
	const relTol = 5e-3
	if maxRel > relTol {
		t.Errorf("trajectory parity maxRel %.3e > %.1e", maxRel, relTol)
	}

	// Residency contract.
	if acc.WeightUploads != 1 {
		t.Errorf("matrix-weight uploads = %d, want 1 (weights uploaded once)", acc.WeightUploads)
	}
	if acc.MomentumUploads != 1 {
		t.Errorf("momentum uploads = %d, want 1 (momentum uploaded once)", acc.MomentumUploads)
	}
	if acc.MomentumReads != 0 {
		t.Errorf("per-step momentum reads = %d, want 0 (momentum resident, not round-tripped)", acc.MomentumReads)
	}
	// No per-step weight movement.
	if acc.WeightReads != 0 {
		t.Errorf("per-step weight reads = %d, want 0 (weights resident, no per-step read-back)", acc.WeightReads)
	}
	if acc.FinalWeightRead != 1 {
		t.Errorf("final weight read-backs = %d, want 1 (single checkpoint read)", acc.FinalWeightRead)
	}
	if acc.GradUploads != testSteps {
		t.Errorf("per-step grad uploads = %d, want %d (grads recomputed each step)", acc.GradUploads, testSteps)
	}
	t.Logf("residency: matrixElems=%d weightUploads=%d momentumUploads=%d momentumReads=%d gradUploads=%d weightReads=%d finalWeightRead=%d",
		acc.MatrixElems, acc.WeightUploads, acc.MomentumUploads, acc.MomentumReads, acc.GradUploads, acc.WeightReads, acc.FinalWeightRead)
}
