package devicemath

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
)

func TestGatedMLPBackwardResidentBindings(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	for _, shape := range [][3]int{{1, 3, 5}, {4, 5, 6}} {
		for _, resident := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/resident=%t", shape, resident), func(t *testing.T) {
				rows, width, inter := shape[0], shape[1], shape[2]
				rng := rand.New(rand.NewSource(66))
				x := randSlice(rng, rows*width)
				weights := [][]float32{randSlice(rng, width*inter), randSlice(rng, width*inter), randSlice(rng, width*inter)}
				g, a, u, h := hostGatedMLPT32(x, weights[0], weights[1], weights[2], rows, width, inter)
				mw := mlpMatW{hostW(weights[0]), hostW(weights[1]), hostW(weights[2])}
				bindings := []*linWeight{&mw.gate, &mw.up, &mw.down}
				var owned []driver.DevicePtr
				defer func() {
					if err := FreeResident(worker, owned...); err != nil {
						t.Error(err)
					}
				}()
				if resident {
					for index, binding := range bindings {
						weight, err := AllocResidentF32(worker, len(weights[index]), weights[index])
						if err != nil {
							t.Fatal(err)
						}
						owned = append(owned, weight)
						gradient, err := AllocResidentF32(worker, len(weights[index]), nil)
						if err != nil {
							t.Fatal(err)
						}
						owned = append(owned, gradient)
						*binding = linWeight{dev: weight, gradient: gradient}
					}
				}
				for range 2 {
					incoming := randSlice(rng, len(x))
					before, err := worker.ExecutionStats(t.Context())
					if err != nil {
						t.Fatal(err)
					}
					got, err := gatedMLPBackwardTW(worker, x, mw, g, a, u, h, incoming, rows, width, inter)
					if err != nil {
						t.Fatal(err)
					}
					after, err := worker.ExecutionStats(t.Context())
					if err != nil {
						t.Fatal(err)
					}
					upload, download := uint64(2*len(x)+4*len(g))*f32Bytes, uint64(len(x))*f32Bytes
					grads := [][]float32{got.DWGate, got.DWUp, got.DWDown}
					for index, binding := range bindings {
						if resident {
							if grads[index] != nil {
								t.Fatal("resident gradient returned a host copy")
							}
							grads[index] = make([]float32, len(weights[index]))
							if err := ReadResident(worker, binding.gradient, ResidentSlice{Data: grads[index]}); err != nil {
								t.Fatal(err)
							}
							unchanged := make([]float32, len(weights[index]))
							if err := ReadResident(worker, binding.dev, ResidentSlice{Data: unchanged}); err != nil {
								t.Fatal(err)
							}
							if !slices.Equal(unchanged, weights[index]) {
								t.Fatal("borrowed weight changed")
							}
						} else {
							upload += uint64(len(weights[index])) * f32Bytes
							download += uint64(len(weights[index])) * f32Bytes
						}
					}
					// Same finite-difference authority and tolerance as the existing MLP tests.
					allGradients := append([][]float32{got.DX}, grads...)
					for index, values := range append([][]float32{x}, weights...) {
						gradient := allGradients[index]
						for element, original := range values {
							const epsilon = 1e-2
							values[element] = original + epsilon
							plus := gatedMLPLossTF64(x, weights[0], weights[1], weights[2], incoming, rows, width, inter)
							values[element] = original - epsilon
							minus := gatedMLPLossTF64(x, weights[0], weights[1], weights[2], incoming, rows, width, inter)
							values[element] = original
							if difference := math.Abs((plus-minus)/(2*epsilon) - float64(gradient[element])); math.IsNaN(difference) || difference > 3e-3 {
								t.Fatalf("gradient %d element %d difference %g", index, element, difference)
							}
						}
					}
					t.Logf("HtoD=%d DtoH=%d synchronizations=%d", after.HostToDeviceBytes-before.HostToDeviceBytes, after.DeviceToHostBytes-before.DeviceToHostBytes, after.StreamSynchronizations-before.StreamSynchronizations)
					if after.HostToDeviceBytes-before.HostToDeviceBytes != upload || after.DeviceToHostBytes-before.DeviceToHostBytes != download || after.StreamSynchronizations-before.StreamSynchronizations != 1 {
						t.Errorf("MLP VJP crossed an internal host boundary; want HtoD=%d DtoH=%d and one synchronization", upload, download)
					}
				}
			})
		}
	}
	stats, err := worker.MemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if stats.CurrentBytes != 0 {
		t.Fatalf("MLP retained %d bytes", stats.CurrentBytes)
	}
}

// Invalid bindings must fail before a worker is used or a borrowed buffer is written.
func TestGatedMLPResidentRefusals(t *testing.T) {
	for name, mutate := range map[string]func(*mlpMatW){
		"absent":             func(w *mlpMatW) { w.gate = linWeight{} },
		"host-shape":         func(w *mlpMatW) { w.gate = hostW([]float32{1, 2}) },
		"self-alias":         func(w *mlpMatW) { w.gate.gradient = w.gate.dev },
		"cross-weight-alias": func(w *mlpMatW) { w.up.gradient = w.gate.dev },
		"duplicate-gradient": func(w *mlpMatW) { w.up.gradient = w.gate.gradient },
	} {
		t.Run(name, func(t *testing.T) {
			w := mlpMatW{linWeight{dev: 1, gradient: 11}, linWeight{dev: 2, gradient: 12}, linWeight{dev: 3, gradient: 13}}
			mutate(&w)
			one := []float32{1}
			if _, err := gatedMLPBackwardTW(nil, one, w, one, one, one, one, one, 1, 1, 1); err == nil {
				t.Fatal("invalid binding accepted")
			}
		})
	}
}
