package sampling

// splitMixSource: small deterministic rand.Source64 whose complete state
// one uint64, making exact sampler continuation serializable
type splitMixSource struct {
	state uint64
}

func (s *splitMixSource) Seed(seed int64) {
	s.state = uint64(seed)
}

func (s *splitMixSource) Uint64() uint64 {
	s.state += 0x9e3779b97f4a7c15
	value := s.state
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func (s *splitMixSource) Int63() int64 {
	return int64(s.Uint64() >> 1)
}
