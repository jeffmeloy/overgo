package devicemath

import (
	"math/rand"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestLinearBackwardResidentGradient(t *testing.T) {
	cudatest.Require(t)
	weights := HybridLayerResidentMatrices{MLPGate: 1, MLPUp: 2, MLPDown: 3, AttnWq: 4, AttnWk: 5, AttnWv: 6, AttnWo: 7}
	gradients := HybridLayerResidentMatrices{MLPGate: 11, MLPUp: 12, MLPDown: 13, AttnWq: 14, AttnWk: 15, AttnWv: 16, AttnWo: 17}
	if _, err := weights.withGradients(gradients); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*HybridLayerResidentMatrices){
		func(views *HybridLayerResidentMatrices) { views.IsLinear = true },
		func(views *HybridLayerResidentMatrices) { views.AttnWv = 0 },
		func(views *HybridLayerResidentMatrices) { views.AttnWv = weights.MLPGate },
		func(views *HybridLayerResidentMatrices) { views.AttnWv = views.MLPGate },
	} {
		invalid := gradients
		mutate(&invalid)
		if _, err := weights.withGradients(invalid); err == nil {
			t.Fatal("invalid resident matrix binding accepted")
		}
	}
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	for _, shape := range [][3]int{{1, 3, 5}, {7, 8, 3}} {
		rows, in, out := shape[0], shape[1], shape[2]
		rng := rand.New(rand.NewSource(77))
		x, weight := randSlice(rng, rows*in), randSlice(rng, in*out)
		dWeight := testcheck.Must(AllocResidentF32(worker, len(weight), weight))(t)
		dGradient := testcheck.Must(AllocResidentF32(worker, len(weight), nil))(t)
		for range 2 {
			incoming := randSlice(rng, rows*out)
			wantX, wantWeight := make([]float32, len(x)), make([]float32, len(weight))
			hostmath.LinearBackward(wantX, wantWeight, nil, x, weight, incoming, rows, in, out, false)
			before := testcheck.Must(worker.ExecutionStats(t.Context()))(t)
			gotX, gotWeight, err := linearBackwardTW(worker, x, linWeight{dev: dWeight, gradient: dGradient}, incoming, rows, in, out)
			testcheck.NoError(t, err)
			after := testcheck.Must(worker.ExecutionStats(t.Context()))(t)
			if gotWeight != nil || after.DeviceToHostBytes-before.DeviceToHostBytes != uint64(len(x))*f32Bytes {
				t.Fatal("matrix VJP crossed the host boundary")
			}
			if after.HostToDeviceBytes-before.HostToDeviceBytes != uint64(len(x)+len(incoming))*f32Bytes {
				t.Fatal("resident weight was uploaded")
			}
			gotWeight = make([]float32, len(weight))
			testcheck.NoError(t, ReadResident(worker, dGradient, ResidentSlice{Data: gotWeight}))
			for _, values := range [][2][]float32{{gotX, wantX}, {gotWeight, wantWeight}} {
				if difference := testutil.MaxAbsDiff(values[0], values[1]); difference > 1e-4 {
					t.Fatalf("shape=%v gradient error=%g", shape, difference)
				}
			}
			gotWeight = make([]float32, len(weight))
			testcheck.NoError(t, ReadResident(worker, dWeight, ResidentSlice{Data: gotWeight}))
			if !slices.Equal(gotWeight, weight) {
				t.Fatal("borrowed matrix weights changed")
			}
			if _, _, err := linearBackwardTW(worker, x, linWeight{dev: dWeight, gradient: dWeight}, incoming, rows, in, out); err == nil {
				t.Fatal("aliased weight and gradient accepted")
			}
		}
		testcheck.NoError(t, FreeResident(worker, dWeight, dGradient))
	}
	stats := testcheck.Must(worker.MemoryStats(t.Context()))(t)
	if stats.CurrentBytes != 0 {
		t.Fatalf("VJP retained %d bytes after caller release", stats.CurrentBytes)
	}
}
