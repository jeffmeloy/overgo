package executor

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func PackedCopy(values []DeviceValue, storage dtype.Type) (DeviceCopy, error) {
	shape, bytes, err := packedLayout(values, storage)
	if err != nil {
		return DeviceCopy{}, err
	}
	segments := make([]DeviceCopySegment, len(values))
	for index, value := range values {
		segments[index] = DeviceCopySegment{Source: value.Pointer, Bytes: bytes}
	}
	return DeviceCopy{Shape: shape, Storage: storage, Segments: segments}, nil
}

func PackedView(values []DeviceValue, storage dtype.Type) (DeviceValue, bool) {
	shape, bytes, err := packedLayout(values, storage)
	if err != nil {
		return DeviceValue{}, false
	}
	last := uint64(len(values) - tensor.SingletonExtent)
	if last > math.MaxUint64/bytes ||
		uint64(values[0].Pointer) > math.MaxUint64-last*bytes {
		return DeviceValue{}, false
	}
	for index, value := range values {
		want := values[0].Pointer + driver.DevicePtr(uint64(index)*bytes)
		if value.Pointer != want {
			return DeviceValue{}, false
		}
	}
	return DeviceValue{Pointer: values[0].Pointer, Shape: shape}, true
}

func packedLayout(values []DeviceValue, storage dtype.Type) (tensor.Shape, uint64, error) {
	if len(values) <= tensor.SingletonExtent {
		return tensor.Shape{}, 0, errors.New("packed device values require multiple inputs")
	}
	base := values[0].Shape
	shape, err := tensor.PackBatchShape(base, uint64(len(values)))
	bytes, byteErr := base.Bytes(storage)
	if err != nil || byteErr != nil || bytes == 0 {
		return tensor.Shape{}, 0, errors.New("packed device shape is invalid")
	}
	for _, value := range values {
		if value.Pointer == 0 || !value.Shape.Equal(base) {
			return tensor.Shape{}, 0, errors.New("packed device values differ")
		}
	}
	return shape, bytes, nil
}

func SplitPackedValue(
	packed DeviceValue, template tensor.Shape, storage dtype.Type, sequence uint64,
) (DeviceValue, error) {
	shape := packed.Shape
	switch {
	case shape.Rank == template.Rank+tensor.SingletonExtent:
		shape.Rank--
		shape.Dims[shape.Rank] = tensor.SingletonExtent
	case shape.Rank == template.Rank && template.Rank == tensor.MaxDimensions:
		shape.Dims[shape.Rank-tensor.SingletonExtent] = tensor.SingletonExtent
	default:
		return DeviceValue{}, errors.New("packed device shape differs from template")
	}
	view, err := packed.SliceLastAxis(storage, sequence, tensor.SingletonExtent)
	if err != nil {
		return DeviceValue{}, err
	}
	view.Shape = shape
	return view, nil
}

func (v DeviceValue) SliceLastAxis(storage dtype.Type, start, count uint64) (DeviceValue, error) {
	if v.Shape.Rank == 0 {
		return DeviceValue{}, errors.New("device value has no slice axis")
	}
	axis := v.Shape.Rank - 1
	length := v.Shape.Dims[axis]
	end, ok := checked.Add64(start, count)
	if !ok || end > length {
		return DeviceValue{}, errors.New("device value slice exceeds last axis")
	}
	unit := v.Shape
	unit.Dims[axis] = 1
	stride, err := unit.Bytes(storage)
	if err != nil {
		return DeviceValue{}, err
	}
	offset, ok := checked.Mul64(start, stride)
	if !ok || uint64(v.Pointer) > math.MaxUint64-offset {
		return DeviceValue{}, errors.New("device value slice pointer overflows")
	}
	if v.CapacityBytes != 0 {
		viewBytes, ok := checked.Mul64(count, stride)
		if !ok || offset > v.CapacityBytes || viewBytes > v.CapacityBytes-offset {
			return DeviceValue{}, errors.New("device value slice exceeds capacity")
		}
		v.CapacityBytes -= offset
	}
	v.Pointer += driver.DevicePtr(offset)
	v.Shape.Dims[axis] = count
	return v, nil
}

func (v DeviceValue) SliceLastAxisIfExtent(
	storage dtype.Type, start, count, extent uint64,
) (DeviceValue, error) {
	actual, _, valid := tensor.FinalExtent(v.Shape)
	if !valid || actual != extent {
		return v, nil
	}
	return v.SliceLastAxis(storage, start, count)
}

func (v DeviceValue) SliceLastAxisExactExtent(
	storage dtype.Type, start, count, extent uint64,
) (DeviceValue, error) {
	actual, _, valid := tensor.FinalExtent(v.Shape)
	if !valid || actual != extent {
		return DeviceValue{}, errors.New("device value final extent differs")
	}
	return v.SliceLastAxis(storage, start, count)
}

func (v DeviceValue) Copy(storage dtype.Type) (DeviceCopy, error) {
	bytes, err := v.Shape.Bytes(storage)
	if err != nil {
		return DeviceCopy{}, err
	}
	return DeviceCopy{Shape: v.Shape, Storage: storage, Segments: []DeviceCopySegment{{Source: v.Pointer, Bytes: bytes}}}, nil
}

func (v DeviceValue) CopyWithoutLastAxisRange(
	storage dtype.Type, keep, discard, expectedExtent uint64,
) (DeviceCopy, error) {
	extent, _, valid := tensor.FinalExtent(v.Shape)
	suffixStart, validRange := checked.Add64(keep, discard)
	if !valid || extent != expectedExtent || !validRange || suffixStart > extent {
		return DeviceCopy{}, errors.New("device value removal range overflows")
	}
	shape, valid := tensor.WithTrailingExtent(v.Shape, extent-discard)
	if !valid {
		return DeviceCopy{}, errors.New("device value removal shape is invalid")
	}
	segments := make([]DeviceCopySegment, 0, tensor.PairedExtent)
	if keep != tensor.FirstOffset {
		segment, err := v.copySegment(storage, tensor.FirstOffset, keep)
		if err != nil {
			return DeviceCopy{}, err
		}
		segments = append(segments, segment)
	}
	if suffix := extent - suffixStart; suffix != tensor.FirstOffset {
		segment, err := v.copySegment(storage, suffixStart, suffix)
		if err != nil {
			return DeviceCopy{}, err
		}
		segments = append(segments, segment)
	}
	return DeviceCopy{Shape: shape, Storage: storage, Segments: segments}, nil
}

func (v DeviceValue) copySegment(storage dtype.Type, start, count uint64) (DeviceCopySegment, error) {
	view, err := v.SliceLastAxis(storage, start, count)
	if err != nil {
		return DeviceCopySegment{}, err
	}
	bytes, err := view.Shape.Bytes(storage)
	return DeviceCopySegment{Source: view.Pointer, Bytes: bytes}, err
}

// ReadDeviceValues copies F32 device values to the host in one worker transaction.
//   - each value's shape names the elements read from its pointer
//   - a zero pointer, or a shape larger than a known capacity, is refused before any copy.
func (e *Executor) ReadDeviceValues(ctx context.Context, values []DeviceValue) ([][]float32, error) {
	host := make([][]float32, len(values))
	for index, value := range values {
		elements, err := transferElements(value)
		if err != nil {
			return nil, fmt.Errorf("CUDA device read %d: %w", index, err)
		}
		host[index] = make([]float32, elements)
	}
	err := e.transfer(ctx, func(state *device.State) error {
		for index, value := range values {
			if err := state.Driver.MemcpyDtoH(driver.Bytes(host[index]), value.Pointer); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return host, nil
}

// WriteDeviceValues copies host F32 data into device values in one worker transaction.
//   - data[i] holds exactly the elements values[i]'s shape names
//   - a zero pointer, a count mismatch, or a shape larger than a known capacity is refused before any copy.
func (e *Executor) WriteDeviceValues(ctx context.Context, values []DeviceValue, data [][]float32) error {
	if len(values) != len(data) {
		return fmt.Errorf("CUDA device write has %d values and %d host slices", len(values), len(data))
	}
	for index, value := range values {
		elements, err := transferElements(value)
		if err != nil {
			return fmt.Errorf("CUDA device write %d: %w", index, err)
		}
		if len(data[index]) != elements {
			return fmt.Errorf("CUDA device write %d has %d host values, its shape holds %d", index, len(data[index]), elements)
		}
	}
	return e.transfer(ctx, func(state *device.State) error {
		for index, value := range values {
			if err := state.Driver.MemcpyHtoD(value.Pointer, driver.Bytes(data[index])); err != nil {
				return err
			}
		}
		return nil
	})
}

// DeviceBytes is a host byte range for a device address: a table of integers or codes, which DeviceValue's F32
// transfers do not describe.
type DeviceBytes struct {
	Pointer driver.DevicePtr
	Data    []byte
}

// WriteDeviceBytes copies host bytes to device addresses in one worker transaction; a zero pointer or an empty range is
// refused before any copy.
func (e *Executor) WriteDeviceBytes(ctx context.Context, writes []DeviceBytes) error {
	for index, write := range writes {
		if write.Pointer == 0 || len(write.Data) == 0 {
			return fmt.Errorf("CUDA device byte write %d is empty", index)
		}
	}
	return e.transfer(ctx, func(state *device.State) error {
		for _, write := range writes {
			if err := state.Driver.MemcpyHtoD(write.Pointer, write.Data); err != nil {
				return err
			}
		}
		return nil
	})
}

// ReadDeviceBytes copies device bytes into host ranges in one worker transaction, each read filling its Data from its
// Pointer; a zero pointer or an empty range is refused before any copy.
func (e *Executor) ReadDeviceBytes(ctx context.Context, reads []DeviceBytes) error {
	for index, read := range reads {
		if read.Pointer == 0 || len(read.Data) == 0 {
			return fmt.Errorf("CUDA device byte read %d is empty", index)
		}
	}
	return e.transfer(ctx, func(state *device.State) error {
		for _, read := range reads {
			if err := state.Driver.MemcpyDtoH(read.Data, read.Pointer); err != nil {
				return err
			}
		}
		return nil
	})
}

// PinnedHostBuffer is pinned host memory mapped into the device: the host reads and writes Host, kernels read and
// write Device over the bus, as a streamed QSA layer's host copy is (Strata's kv_stream.hpp).
type PinnedHostBuffer struct {
	executor *Executor
	Host     []byte
	Device   driver.DevicePtr
}

// AllocatePinnedHost pins bytes of host memory mapped into the device.
func (e *Executor) AllocatePinnedHost(ctx context.Context, bytes uint64) (*PinnedHostBuffer, error) {
	if bytes == 0 {
		return nil, errors.New("CUDA pinned host request is empty")
	}
	buffer := &PinnedHostBuffer{executor: e}
	err := e.transfer(ctx, func(state *device.State) error {
		var err error
		buffer.Host, buffer.Device, err = state.Driver.MemHostAlloc(bytes)
		return err
	})
	if err != nil {
		return nil, err
	}
	return buffer, nil
}

// Release unpins the buffer once the device is done with it; a released buffer releases nothing again.
func (b *PinnedHostBuffer) Release(ctx context.Context) error {
	if b == nil || b.Host == nil {
		return nil
	}
	err := b.executor.transfer(ctx, func(state *device.State) error {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return state.Driver.MemFreeHost(b.Host)
	})
	if err == nil {
		b.Host, b.Device = nil, 0
	}
	return err
}

// DeviceRangeCopy is a device-to-device copy of Bytes from Source to Target.
type DeviceRangeCopy struct {
	Target, Source driver.DevicePtr
	Bytes          uint64
}

// CopyDeviceRanges copies device ranges into existing device memory in one worker transaction, as a state that grows
// or forks copies the prefix it keeps; a zero pointer, an empty range or a target overlapping its source is refused
// before any copy.
func (e *Executor) CopyDeviceRanges(ctx context.Context, copies []DeviceRangeCopy) error {
	for index, copySpec := range copies {
		source, sourceErr := newDeviceAddressRange(copySpec.Source, copySpec.Bytes)
		target, targetErr := newDeviceAddressRange(copySpec.Target, copySpec.Bytes)
		if copySpec.Bytes == 0 || sourceErr != nil || targetErr != nil || deviceRangesOverlap(source, target) {
			return fmt.Errorf("CUDA device range copy %d is empty, invalid or overlaps its source", index)
		}
	}
	return e.transfer(ctx, func(state *device.State) error {
		for _, copySpec := range copies {
			if err := state.Driver.MemcpyDtoD(copySpec.Target, copySpec.Source, copySpec.Bytes); err != nil {
				return err
			}
		}
		return nil
	})
}

func transferElements(value DeviceValue) (int, error) {
	if value.Pointer == 0 {
		return 0, errors.New("device value pointer is zero")
	}
	bytes, err := value.Shape.Bytes(dtype.F32)
	if err != nil {
		return 0, err
	}
	if value.CapacityBytes != 0 && bytes > value.CapacityBytes {
		return 0, fmt.Errorf("device value shape holds %d bytes, its capacity is %d", bytes, value.CapacityBytes)
	}
	elements, err := value.Shape.Elements()
	if err != nil {
		return 0, err
	}
	count, ok := checked.Int(elements)
	if !ok {
		return 0, fmt.Errorf("device value holds %d elements, more than the host addresses", elements)
	}
	return count, nil
}

func (e *Executor) transfer(ctx context.Context, copies func(*device.State) error) error {
	if e == nil {
		return errors.New("CUDA executor is closed")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return errors.New("CUDA executor is closed")
	}
	return e.worker.Do(ctx, copies)
}
