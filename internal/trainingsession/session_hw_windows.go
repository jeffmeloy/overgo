//go:build windows

package trainingsession

import (
	"context"

	"overgo/internal/cuda/device"
)

// sessionSampler holds one device worker for the whole supervised run so
// samples are driver-reported device-global used bytes (cuMemGetInfo total
// minus free) -- visible across contexts, so the training worker's residency
// is measured rather than the sampler's own allocator accounting. A host
// without a usable device reports no sampler rather than failing the run.
type sessionSampler struct {
	worker *device.Worker
}

func newSessionSampler() *sessionSampler {
	worker, err := device.New(0)
	if err != nil {
		return nil
	}
	return &sessionSampler{worker: worker}
}

// sample returns device-global used bytes.
func (s *sessionSampler) sample() (uint64, bool) {
	if s == nil {
		return 0, false
	}
	var used uint64
	err := s.worker.Do(context.Background(), func(state *device.State) error {
		free, total, err := state.Driver.MemInfo()
		if err != nil {
			return err
		}
		used = total - free
		return nil
	})
	if err != nil {
		return 0, false
	}
	return used, true
}

func (s *sessionSampler) close() {
	if s != nil {
		s.worker.Close()
	}
}
