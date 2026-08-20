//go:build !windows

package trainingsession

// sessionSampler: no CUDA device off Windows; observations carry phase walls
// without hardware samples.
type sessionSampler struct{}

func newSessionSampler() *sessionSampler { return nil }

func (s *sessionSampler) sample() (uint64, bool) { return 0, false }

func (s *sessionSampler) close() {}
