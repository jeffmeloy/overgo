//go:build !windows

package trainingworkflow

type sessionSampler struct{}

func newSessionSampler() *sessionSampler { return nil }

func (s *sessionSampler) sample() (used uint64, ok bool) { return }

func (s *sessionSampler) close() {}
