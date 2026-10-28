package executor

import (
	"context"
	"errors"

	"overgo/internal/cuda/device"
)

// probe_pcie_h2d_gbps (Strata at 296b59d2, MIT: src/program/generate.cpp), the start-up probe of the link the round's
// PCIe share crosses: 256 MiB of pinned host memory copied to the card four times back to back after a warm-up copy, each
// burst timed by the events between them, the best burst's rate.
const (
	strataProbeBytes  = 256 << 20
	strataProbeBursts = 4
	// strataProbeFloorMS is the shortest burst read as a rate (0.01 ms); strataMillisecond and strataGB the units.
	strataProbeFloorMS = 0.01
	strataMillisecond  = 1e-3
	strataGB           = 1e9
)

// ProbeLinkGBps answers the host-to-card bandwidth in GB/s, or a negative value where nothing could be measured.
func (e *Executor) ProbeLinkGBps(ctx context.Context) (float64, error) {
	best := float64(-1)
	err := e.worker.Do(ctx, func(state *device.State) error {
		host, _, err := state.Driver.MemHostAlloc(strataProbeBytes)
		if err != nil {
			return nil
		}
		card, err := e.resources.allocate(state, strataProbeBytes)
		if err != nil {
			return state.Driver.MemFreeHost(host)
		}
		clear(host) // fault the pages in before timing
		// the warm-up: the context up, the copy engine primed
		warmed := state.Driver.MemcpyHtoDAsync(card, host, state.Stream)
		times, timed := state.Driver.TimeCopiesHtoD(card, host, strataProbeBursts, state.Stream)
		if warmed == nil && timed == nil {
			for _, ms := range times {
				if ms > strataProbeFloorMS {
					best = max(best, strataProbeBytes/(float64(ms)*strataMillisecond)/strataGB)
				}
			}
		}
		return errors.Join(state.Driver.MemFree(card), state.Driver.MemFreeHost(host))
	})
	return best, err
}
