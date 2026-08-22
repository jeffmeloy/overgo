//go:build windows

package trainingworkflow

import (
	"context"

	"overgo/internal/cuda/device"
)

type sessionSampler struct {
	worker *device.Worker
}

func newSessionSampler() *sessionSampler {
	worker, err := newDeviceWorker()
	if err != nil {
		return nil
	}
	return &sessionSampler{worker: worker}
}

func (s *sessionSampler) sample() (used uint64, ok bool) {
	if s == nil {
		return used, ok
	}
	err := s.worker.Do(context.Background(), func(state *device.State) error {
		free, total, err := state.Driver.MemInfo()
		if err != nil {
			return err
		}
		used = total - free
		return nil
	})
	if err != nil {
		return used, ok
	}
	return used, true
}

func (s *sessionSampler) close() {
	if s != nil {
		s.worker.Close()
	}
}
