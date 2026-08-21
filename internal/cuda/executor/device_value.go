package executor

import (
	"errors"
	"math"

	"overgo/internal/checked"
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
	return DeviceCopy{Shape: shape, Segments: segments}, nil
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
	return DeviceCopy{Shape: v.Shape, Segments: []DeviceCopySegment{{Source: v.Pointer, Bytes: bytes}}}, nil
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
	return DeviceCopy{Shape: shape, Segments: segments}, nil
}

func (v DeviceValue) copySegment(storage dtype.Type, start, count uint64) (DeviceCopySegment, error) {
	view, err := v.SliceLastAxis(storage, start, count)
	if err != nil {
		return DeviceCopySegment{}, err
	}
	bytes, err := view.Shape.Bytes(storage)
	return DeviceCopySegment{Source: view.Pointer, Bytes: bytes}, err
}
