package hfgguf

import (
	"errors"
	"fmt"
	"io"
)

// transformedPayload owns only the tensor its consumer is reading.
// Completion releases the payload even when the consumer reads no trailing EOF.
// Source and transform errors remain terminal for this reader.
type transformedPayload struct {
	source      io.Reader
	elementSize uint64
	transforms  []TensorTransform
	name        string
	data        []byte
	err         error
}

// Read applies the declared transformations on demand and releases consumed storage.
func (r *transformedPayload) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if r.source != nil {
		r.data, r.err = io.ReadAll(r.source)
		r.source = nil
		if r.err == nil && (r.elementSize == 0 || uint64(len(r.data))%r.elementSize != 0) {
			r.err = errors.New("payload size disagrees with element size")
		}
		for _, transform := range r.transforms {
			if r.err != nil {
				break
			}
			r.err = transform.apply(r.data, r.elementSize)
		}
		r.transforms = nil
		if r.err != nil {
			r.data = nil
			r.err = fmt.Errorf("HF/GGUF adapter: tensor %q: %w", r.name, r.err)
		}
	}
	if r.err != nil {
		return 0, r.err
	}
	n := copy(p, r.data)
	r.data = r.data[n:]
	if len(r.data) == 0 {
		r.data = nil
		r.err = io.EOF
		if n == 0 {
			return 0, io.EOF
		}
	}
	return n, nil
}
