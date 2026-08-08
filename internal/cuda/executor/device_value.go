package executor

import (
	"errors"
	"math"

	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor/dtype"
)

// SliceLastAxis: contiguous last-axis device view.
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
