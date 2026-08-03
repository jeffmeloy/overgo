package driver

import "unsafe"

type hostScalar interface {
	~float32 | ~uint32
}

// Bytes: zero-copy host scalar view.
func Bytes[T hostScalar](values []T) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*int(unsafe.Sizeof(values[0])))
}
