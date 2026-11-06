//go:build windows

package latentvideo

import (
	"maps"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/processmeasure"
	"overgo/internal/testcheck"
)

// TestADenoiserLoadHoldsOneTensorOnTheHost: a Wan denoiser's session uploads its weights a tensor at a time from the
// open file, decoding each as it uploads it. The opened set holds decoded only the timestep conditioning's, which compile
// reads, and the process's peak working set grows by less than a quarter of the weights while the session is made, where
// decoding every tensor before the upload grew it by 5,440,901,120 bytes for 5,641,371,904 of weights.
func TestADenoiserLoadHoldsOneTensorOnTheHost(t *testing.T) {
	cudatest.Require(t)
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	dir := wanModelDir(t)
	profile := testcheck.Must(ResolveProfile(dir))(t)
	config := testcheck.Must(LoadDenoiserConfig(dir, profile.Policy))(t)
	geometry := testcheck.Must(config.CompileLatentGeometry(geometryFrames, geometrySide, geometrySide))(t)
	before := testcheck.Must(processmeasure.SelfPeakWorkingSet())(t)
	weights := testcheck.Must(openDenoiserWeights(dir, config))(t)
	defer weights.Close()
	if held := slices.Sorted(maps.Keys(weights.values)); !slices.Equal(held, slices.Sorted(slices.Values(timestepTensors))) {
		t.Fatalf("the opened weights hold %v decoded, want the timestep conditioning's alone", held)
	}
	program := testcheck.Must(CompileDenoiserProgramPrecision(config, weights, geometry, DenoiserPrecision{}))(t)
	session := testcheck.Must(NewDenoiserCUDASession(program, 0))(t)
	defer session.Close()
	after := testcheck.Must(processmeasure.SelfPeakWorkingSet())(t)
	t.Logf("the session's load grew the process's peak working set by %d bytes for %d bytes of weights", after-before, weights.Bytes)
	if growth := after - before; growth*quarters > uint64(weights.Bytes) {
		t.Fatalf("the load grew the peak working set by %d bytes, more than a quarter of the %d bytes of weights", growth, weights.Bytes)
	}
}

// The smallest clip the test's session is compiled for, and the share of the weights its load may grow the host by.
const (
	geometryFrames = 5
	geometrySide   = 64
	quarters       = 4
)
