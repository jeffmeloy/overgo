//go:build windows

package executor

import (
	"encoding/binary"
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor/dtype"
)

// TestBF16PackKeepsNaN: the device's pack keeps every NaN a NaN, the GPU's default ones and those whose mantissa lies below
// BF16's among them, and agrees with the host's on every input, so a numeric fault cannot read as a plausible value.
func TestBF16PackKeepsNaN(t *testing.T) {
	nan := []uint32{0x7fffffff, 0xffffffff, 0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001, 0x7f80ffff, 0xff80ffff}
	finite := []uint32{0x7f800000, 0xff800000, 0, 0x80000000, 0x3f800000, 0xbfc00000, 0x3f808000, 0x3f818000, 0x7f7fffff}
	inputs := slices.Concat(nan, finite)
	data := make([]byte, 0, len(inputs)*binary.Size(uint32(0)))
	for _, input := range inputs {
		data = binary.LittleEndian.AppendUint32(data, input)
	}
	cuda := newFixtureExecutor(t)
	buffers := newStrataDevice(t, cuda)
	input, output := buffers.upload(t, data), buffers.zeros(t, len(inputs)*binary.Size(uint16(0)))
	count := uint32(len(inputs))
	buffers.run(t, func(state *device.State, functions functionSet) error {
		return launch1DABI(state, functions[kernelF32ToBf16], count, &input, &output, &count)
	})
	packed := buffers.read(t, output, len(inputs)*binary.Size(uint16(0)))
	for index, bits := range inputs {
		got := binary.LittleEndian.Uint16(packed[index*binary.Size(uint16(0)):])
		if want := dtype.Float32ToBF16(math.Float32frombits(bits)); got != want {
			t.Errorf("%#08x packs to %#04x on the device and %#04x on the host", bits, got, want)
		}
		if isNaN := math.IsNaN(float64(dtype.BF16ToFloat32(got))); isNaN != (index < len(nan)) {
			t.Errorf("%#08x packs to %#04x, NaN %t", bits, got, isNaN)
		}
	}
}
