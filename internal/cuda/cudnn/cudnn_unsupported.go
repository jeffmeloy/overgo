//go:build !windows

package cudnn

import "overgo/internal/cuda/driver"

// Library is cuDNN where it can be loaded, which is on Windows.
type Library struct{}

// Convolution is a planned forward convolution.
type Convolution struct{ Workspace uint64 }

// Open loads nothing here: ErrAbsent.
func Open(driver.Stream) (*Library, error) { return nil, ErrAbsent }

// Close releases a library, and there is none.
func (l *Library) Close() error { return nil }

// Plan plans nothing without a library: ErrAbsent.
func (l *Library) Plan(Shape, DataType, DataType) (Convolution, error) {
	return Convolution{}, ErrAbsent
}

// Release destroys a plan, and there is none: ErrAbsent.
func (l *Library) Release(Convolution) error { return ErrAbsent }

// Forward convolves nothing without a library: ErrAbsent.
func (l *Library) Forward(Convolution, driver.DevicePtr, driver.DevicePtr, driver.DevicePtr, driver.DevicePtr) error {
	return ErrAbsent
}
