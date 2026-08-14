//go:build windows

package hybridtrain

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/optimizer"
)

// smallHybrid is the explicit, magic-free test model: a 2-layer hybrid stack with
// one full_attention layer (partial rope) and one linear_attention (GDN) layer.
func smallHybrid() StackConfig {
	return StackConfig{
		Types:  []LayerKind{FullAttention, LinearAttention},
		Tokens: 6, Hidden: 16, Inter: 24, Eps: 1e-6,
		Heads: 2, KVHeads: 1, HeadDim: 8, RopeDim: 4, RopeTheta: 10000,
		GDNKeyHeads: 2, GDNValueHeads: 2, GDNHeadDim: 4, GDNConvK: 3,
	}
}

// TestHybridTrainDeviceResidentMatchesHost is the FINAL milestone parity test: it
// runs K steps of the multi-layer hybrid training loop on host (TrainHost, the
// oracle) and on the device with weights + Muon momentum resident (TrainDeviceResident),
// from bit-identical initial weights, and asserts (1) both loss trajectories fall,
// (2) they agree step-by-step within the fp32-device-vs-f64-host tolerance, and
// (3) the residency contract holds -- matrix weights and momentum are uploaded ONCE,
// momentum is never round-tripped per step.
func TestHybridTrainDeviceResidentMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const (
		seed  = int64(20250812)
		steps = 8
	)
	cfg := smallHybrid()
	ocfg := optimizer.Config{BaseLearningRate: 0.02, Momentum: 0.9, Schedule: optimizer.ScheduleConstant}

	host, err := BuildModel(cfg, seed)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := BuildModel(cfg, seed)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("resident matrix params=%d  host-Sign vector params=%d", host.MatrixParamCount(), host.VectorParamCount())

	hostTraj, err := host.TrainHost(steps, ocfg)
	if err != nil {
		t.Fatal(err)
	}
	devTraj, acc, err := dev.TrainDeviceResident(worker, steps, ocfg)
	if err != nil {
		t.Fatal(err)
	}

	// Trajectory parity.
	var maxAbs, maxRel float64
	for i := range hostTraj {
		d := math.Abs(hostTraj[i] - devTraj[i])
		r := d / (math.Abs(hostTraj[i]) + 1e-9)
		if d > maxAbs {
			maxAbs = d
		}
		if r > maxRel {
			maxRel = r
		}
		t.Logf("step %d  host=%.6f  device=%.6f  |diff|=%.3e  rel=%.3e", i, hostTraj[i], devTraj[i], d, r)
	}
	t.Logf("trajectory parity over %d steps: max|diff|=%.3e  maxRel=%.3e", steps, maxAbs, maxRel)

	// Both trajectories must actually train down.
	if !(hostTraj[len(hostTraj)-1] < hostTraj[0]) {
		t.Fatalf("host loss did not decrease: %.6f -> %.6f", hostTraj[0], hostTraj[len(hostTraj)-1])
	}
	if !(devTraj[len(devTraj)-1] < devTraj[0]) {
		t.Fatalf("device loss did not decrease: %.6f -> %.6f", devTraj[0], devTraj[len(devTraj)-1])
	}

	// Parity bar: the loss-trajectory class established by the layer tests
	// (full_attention fp32 causal-softmax ~1e-4 per forward, compounding over layers
	// and steps). Relative loss agreement within 5e-3.
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
	// The crux: ZERO per-step weight motion. Weights are resident across the whole
	// K-step loop and read back to host exactly once, at the final checkpoint.
	if acc.WeightReads != 0 {
		t.Errorf("per-step weight reads = %d, want 0 (weights resident, no per-step read-back)", acc.WeightReads)
	}
	if acc.FinalWeightRead != 1 {
		t.Errorf("final weight read-backs = %d, want 1 (single checkpoint read)", acc.FinalWeightRead)
	}
	if acc.GradUploads != steps {
		t.Errorf("per-step grad uploads = %d, want %d (grads recomputed each step)", acc.GradUploads, steps)
	}
	t.Logf("residency: matrixElems=%d weightUploads=%d momentumUploads=%d momentumReads=%d gradUploads=%d weightReads=%d finalWeightRead=%d",
		acc.MatrixElems, acc.WeightUploads, acc.MomentumUploads, acc.MomentumReads, acc.GradUploads, acc.WeightReads, acc.FinalWeightRead)
}
