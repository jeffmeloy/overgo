package executor

import (
	"context"
	"errors"
	"math"
	"runtime"
	"slices"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/hostexpert"
	"overgo/internal/tensor"
)

// MachineCosts are the costs a model's load measures on this machine, which the executor's choices trade: the link's
// rate copying pinned host memory to the card, and the CPU's time computing an expert in host memory, a call's fixed
// part and each slot's.
type MachineCosts struct {
	LinkBytesPerSecond float64
	CPUFixed, CPUSlot  time.Duration
}

// The measurement's sampling: the copy's bursts (the best is the link's rate), the slots of the larger CPU call, and
// the calls a batch and the batches each size takes (the least batch's mean is its time; the clock reads coarser than
// one call); the sampling trades the measurement's milliseconds at load against its noise. hostExpertSlotsMost bounds
// the slots of a group's expert the CPU takes whatever the costs say, which sizes the staging its outputs reserve
// (hostRowsReserve): it trades that staging against the experts the CPU may take. hostExpertSlotsUnmeasured is the
// limit where nothing was measured.
const (
	machineCopyBursts         = 4
	machineSlotCalls          = 10
	machineSlotBatches        = 3
	hostExpertSlotsMost       = 32
	hostExpertSlotsUnmeasured = hostExpertSlotsMost
)

// hostSlotLimit is the most slots an expert of bytes may have and be computed on the CPU where it lies rather than
// copied to the card: the CPU's time, its call's fixed part and a slot's for each slot, costs less than the expert's
// copy over the link while the slots stay below the limit (the card's own products run beside either). It is at most
// hostExpertSlotsMost, and that where nothing was measured.
func (c MachineCosts) hostSlotLimit(bytes uint64) uint32 {
	if c.LinkBytesPerSecond <= 0 || c.CPUSlot <= 0 {
		return hostExpertSlotsUnmeasured
	}
	copyTime := float64(bytes) / c.LinkBytesPerSecond * float64(time.Second)
	slots := (copyTime - float64(c.CPUFixed)) / float64(c.CPUSlot)
	return uint32(min(max(math.Floor(slots), 0), hostExpertSlotsMost))
}

// SetMachineCosts binds the costs a model's load measured, which the executor's choices then trade.
func (e *Executor) SetMachineCosts(costs MachineCosts) error {
	if e == nil {
		return errors.New("machine costs executor is unavailable")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("machine costs executor is closed")
	}
	e.resources.costs = costs
	if e.resources.blas != nil {
		e.resources.blas.costs = costs
	}
	return nil
}

// MeasureHostExpertCosts measures this machine's costs on one expert of the loaded model in host memory, its gate, up
// and down rows in formats at the hidden and intermediate widths: the expert's bytes copied from pinned memory to the
// card (the best of a few bursts), and the CPU pool computing the expert at one slot and at hostExpertSlotsMost
// (the least batch's mean of each), whose difference over the slots between is a slot's time.
func (e *Executor) MeasureHostExpertCosts(ctx context.Context, expert hostexpert.Expert, formats hostexpert.Formats, hidden, intermediate int) (MachineCosts, error) {
	if !hostexpert.Supported(formats, hidden, intermediate) {
		return MachineCosts{}, errors.New("machine costs: the CPU does not compute the expert's formats")
	}
	var costs MachineCosts
	bytes := uint64(len(expert.Gate) + len(expert.Up) + len(expert.Down))
	if err := e.worker.Do(ctx, func(state *device.State) error {
		host, _, err := state.Driver.MemHostAlloc(bytes)
		if err != nil {
			return err
		}
		card, err := e.resources.allocate(state, bytes)
		if err != nil {
			return errors.Join(err, state.Driver.MemFreeHost(host))
		}
		copy(host, expert.Gate)
		copy(host[len(expert.Gate):], expert.Up)
		copy(host[len(expert.Gate)+len(expert.Up):], expert.Down)
		times, timed := state.Driver.TimeCopiesHtoD(card, host, machineCopyBursts, state.Stream)
		if timed == nil && len(times) != 0 && slices.Min(times) > 0 {
			best := slices.Min(times)
			costs.LinkBytesPerSecond = float64(bytes) / (float64(best) * float64(time.Millisecond) / float64(time.Second))
		}
		return errors.Join(timed, state.Driver.MemFree(card), state.Driver.MemFreeHost(host))
	}); err != nil {
		return MachineCosts{}, err
	}
	pool := hostexpert.NewPool(runtime.NumCPU())
	defer pool.Close()
	call := func(slots int) (time.Duration, error) {
		values, activations, out := make([][]float32, slots), make([][]byte, slots), make([][]float32, slots)
		for row := range slots {
			values[row], activations[row], out[row] = make([]float32, hidden), make([]byte, hostexpert.Q8KBytes(hidden)), make([]float32, hidden)
		}
		if err := pool.QuantizeQ8K(values, activations); err != nil {
			return 0, err
		}
		work := []hostexpert.Expert{{Gate: expert.Gate, Up: expert.Up, Down: expert.Down, Activations: activations, Out: out}}
		least := time.Duration(math.MaxInt64)
		for range machineSlotBatches {
			started := time.Now()
			for range machineSlotCalls {
				if err := pool.Layer(work, formats, hidden, intermediate, hostActivation(tensor.MoEAttributes{Activation: tensor.MoEActivationSiLU})); err != nil {
					return 0, err
				}
			}
			least = min(least, time.Since(started)/machineSlotCalls)
		}
		return least, nil
	}
	one, err := call(1)
	if err != nil {
		return MachineCosts{}, err
	}
	most, err := call(hostExpertSlotsMost)
	if err != nil {
		return MachineCosts{}, err
	}
	costs.CPUSlot = max(most-one, 0) / (hostExpertSlotsMost - 1)
	costs.CPUFixed = max(one-costs.CPUSlot, 0)
	return costs, nil
}
