//go:build windows

// Package sampling_test holds the samplers' steps on the card (internal/samplingcuda) to their host steps here.
package sampling_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/samplingcuda"
	"overgo/internal/testcheck"
)

// TestERSDEStepsOnTheCardAsOnTheHost holds samplingcuda.ERSDEDevice to HostERSDE.Step (moss-step-copies): over an
// eight-step flow schedule from 1 to 0, a step skipped among them, the latent the card steps is the host's after every
// step, bit for bit, from the same latent, predictions and noise, through each order the solver takes (the first from
// σ = 1 and again after the skip, the second, the third three times, and the step to 0), with subnormal values among
// them.
func TestERSDEStepsOnTheCardAsOnTheHost(t *testing.T) {
	cudatest.Require(t)
	sigmas := []float32{1, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.05, 0}
	const count, skipped = 4099, 5
	random := rand.New(rand.NewPCG(20261121, 1))
	draw := func() []float32 {
		values := make([]float32, count)
		for i := range values {
			values[i] = float32(random.NormFloat64())
		}
		values[7], values[11] = math.Float32frombits(3), -math.Float32frombits(0x007fffff)
		return values
	}
	host := testcheck.Must(sampling.NewHostERSDE(sigmas, sampling.ERSDEStages))(t)
	card := testcheck.Must(sampling.NewERSDE(sigmas, sampling.ERSDEStages))(t)
	var stepper samplingcuda.ERSDEDevice
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	bytes := uint64(count) * 4
	var latent, denoised, noise, previous driver.DevicePtr
	var firsts [2]driver.DevicePtr
	testcheck.NoError(t, worker.Do(t.Context(), func(state *device.State) error {
		for _, pointer := range []*driver.DevicePtr{&latent, &denoised, &noise, &previous, &firsts[0], &firsts[1]} {
			allocated, err := state.Driver.MemAlloc(bytes)
			if err != nil {
				return err
			}
			*pointer = allocated
		}
		return nil
	}))
	defer worker.Do(t.Context(), func(state *device.State) error {
		for _, pointer := range []driver.DevicePtr{latent, denoised, noise, previous, firsts[0], firsts[1]} {
			_ = state.Driver.MemFree(pointer)
		}
		return stepper.Close(state)
	})
	x := draw()
	stepped := append([]float32(nil), x...)
	var last []float32
	written, orders := 0, map[int]int{}
	for step := range host.Steps() {
		if step == skipped {
			testcheck.NoError(t, host.Skip())
			testcheck.NoError(t, card.Skip())
			last = nil
			continue
		}
		predicted, fresh := draw(), draw()
		if !host.Draws(step) {
			fresh = nil
		}
		want := testcheck.Must(host.Step(x, predicted, fresh))(t)
		plan := testcheck.Must(card.Plan())(t)
		if plan.ToZero {
			stepped = append([]float32(nil), predicted...)
		} else {
			orders[plan.Order]++
			testcheck.NoError(t, worker.Do(t.Context(), func(state *device.State) error {
				uploads := []struct {
					to     driver.DevicePtr
					values []float32
				}{{latent, stepped}, {denoised, predicted}, {noise, fresh}}
				if last != nil {
					uploads = append(uploads, struct {
						to     driver.DevicePtr
						values []float32
					}{previous, last})
				}
				for _, upload := range uploads {
					if err := state.Driver.MemcpyHtoD(upload.to, driver.Bytes(upload.values)); err != nil {
						return err
					}
				}
				var before driver.DevicePtr
				if last != nil {
					before = previous
				}
				if err := stepper.Step(state, plan, latent, denoised, noise, before, firsts[written], firsts[1-written], count); err != nil {
					return err
				}
				if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
					return err
				}
				return state.Driver.MemcpyDtoH(driver.Bytes(stepped), latent)
			}))
			if plan.Order >= 2 {
				written = 1 - written
			}
		}
		for i := range want {
			if math.Float32bits(stepped[i]) != math.Float32bits(want[i]) {
				t.Fatalf("step %d (order %d): element %d is %g on the card, %g on the host", step, plan.Order, i, stepped[i], want[i])
			}
		}
		x, last = want, predicted
	}
	if orders[1] != 2 || orders[2] != 1 || orders[3] != 3 {
		t.Fatalf("the card took orders %v; the schedule should take the first twice, the second once and the third three times", orders)
	}
}
