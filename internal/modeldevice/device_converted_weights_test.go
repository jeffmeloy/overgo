package modeldevice

import (
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/tensor/dtype"
)

func TestDeviceConvertedWeightsEncodesStorage(t *testing.T) {
	values := []float32{-1, 0, 1}
	f32 := encodeF32(values, 0)
	values[0] = values[len(values)-1]
	if got := binary.LittleEndian.Uint32(f32); got != math.Float32bits(values[0]) {
		t.Fatalf("F32 view starts with %x, want %x", got, math.Float32bits(values[0]))
	}
	bytesPerValue, valid := dtype.BF16.ScalarBytes()
	if !valid {
		t.Fatal("BF16 scalar storage is unavailable")
	}
	bf16 := encodeBF16(values, len(values)*int(bytesPerValue))
	for index, value := range values {
		offset := index * int(bytesPerValue)
		if got, want := binary.LittleEndian.Uint16(bf16[offset:]), dtype.Float32ToBF16(value); got != want {
			t.Fatalf("BF16[%d] = %x, want %x", index, got, want)
		}
	}
}
