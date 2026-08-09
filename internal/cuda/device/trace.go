package device

// LaunchTrace: byte log of one execution's launch configuration. Two
// executions with identical traces issue identical device work except for the
// contents of device-resident parameter buffers, so a retained CUDA graph
// captured under one trace replays exactly under the other.
type LaunchTrace struct {
	// Probe: record only; the caller must not issue device work.
	Probe bool
	data  []byte
}

func (t *LaunchTrace) Reset(probe bool) {
	t.Probe = probe
	t.data = t.data[:0]
}

func (t *LaunchTrace) Words(words ...uint64) {
	for _, word := range words {
		t.data = append(t.data,
			byte(word), byte(word>>8), byte(word>>16), byte(word>>24),
			byte(word>>32), byte(word>>40), byte(word>>48), byte(word>>56),
		)
	}
}

func (t *LaunchTrace) Bytes(raw []byte) {
	t.data = append(t.data, raw...)
}

func (t *LaunchTrace) Data() []byte {
	return t.data
}
