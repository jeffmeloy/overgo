package inference

import (
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type deviceAttentionCacheFixture struct {
	KeyWidth, ValueWidth, Heads, Tokens uint32
	KeyPointer, ValuePointer            driver.DevicePtr
}

func (f deviceAttentionCacheFixture) shape(width, tokens uint32) tensor.Shape {
	return tensor.MustShape(uint64(width), uint64(f.Heads), uint64(tokens))
}

func (f deviceAttentionCacheFixture) value(pointer driver.DevicePtr, width, tokens uint32) executor.DeviceValue {
	return executor.DeviceValue{Pointer: pointer, Shape: f.shape(width, tokens)}
}

func (f deviceAttentionCacheFixture) cache() *deviceKVCache {
	return &deviceKVCache{
		Keys:   []executor.DeviceValue{f.value(f.KeyPointer, f.KeyWidth, f.Tokens)},
		Values: []executor.DeviceValue{f.value(f.ValuePointer, f.ValueWidth, f.Tokens)},
		Tokens: f.Tokens,
	}
}

func (f deviceAttentionCacheFixture) advanced(
	pointer driver.DevicePtr,
	width, tokens uint32,
) driver.DevicePtr {
	stride, err := f.shape(width, 1).Bytes(dtype.F32)
	if err != nil {
		panic(err)
	}
	return pointer + driver.DevicePtr(uint64(tokens)*stride)
}
