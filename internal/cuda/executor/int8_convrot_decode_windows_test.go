//go:build windows

package executor

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestTheCardDecodesConvRotAsTheHost holds the card's int8 ConvRot decode (DecodeInt8ConvRot) to the host's float32
// stages (decodeRowF32), which are comfy_kitchen's into a float32 input's dtype, bit for bit: over rows of the
// MOSS Ultra encoder's widths, 4096 and 12288, and of one group, rows of seeded int8 values and of every value at the
// int8 extremes, which give the largest sums, under scales of full mantissas and both signs from 1e-28, whose stages
// stay normal to their last bit (ops_f32 flushes a subnormal value to zero, which no weight's scale reaches), to 1e30,
// zero, an infinite scale and a NaN, each F32 value of every row is the host's, a NaN where the host's is one. A group
// of any width but 256 is refused.
func TestTheCardDecodesConvRotAsTheHost(t *testing.T) {
	cudatest.Require(t)
	random := rand.New(rand.NewPCG(20261122, 1))
	worker := testcheck.Must(device.New(0))(t)
	defer func() { testcheck.NoError(t, worker.Close()) }()
	runtime := testcheck.Must(NewWithWorker(worker))(t)
	defer func() { testcheck.NoError(t, runtime.Close()) }()
	scales := []float32{0, float32(math.Inf(1)), float32(math.NaN()), -1e30, 1e-28, -3e-28}
	decode := func(stored []byte, rows, inner, group uint32) ([]byte, error) {
		decoded := make([]byte, 4*uint64(rows)*uint64(inner))
		err := worker.Do(t.Context(), func(state *device.State) (err error) {
			output, err := state.Driver.MemAlloc(uint64(len(decoded)))
			if err != nil {
				return err
			}
			defer func() { err = errors.Join(err, state.Driver.MemFree(output)) }()
			if err := runtime.DecodeInt8ConvRot(state, Int8ConvRotDecode{Stored: stored, Rows: rows, Inner: inner, Group: group, Output: output}); err != nil {
				return err
			}
			return state.Driver.MemcpyDtoH(decoded, output)
		})
		return decoded, err
	}
	for _, shape := range []struct{ rows, inner uint32 }{{40, 256}, {48, 4096}, {16, 12288}} {
		t.Run(fmt.Sprintf("%dx%d", shape.rows, shape.inner), func(t *testing.T) {
			rows, inner := uint64(shape.rows), uint64(shape.inner)
			stored := make([]byte, rows*inner+4*rows)
			for row := range rows {
				values := stored[row*inner : (row+1)*inner]
				switch row {
				case 0:
					for index := range values {
						values[index] = 0x80
					}
				case 1:
					for index := range values {
						values[index] = 0x7f
					}
				default:
					for index := range values {
						values[index] = byte(random.Uint32())
					}
				}
				scale := float32(math.Ldexp(1+random.Float64(), random.IntN(80)-40)) * float32(1-2*random.IntN(2))
				if row < uint64(len(scales)) {
					scale = scales[row]
				}
				binary.LittleEndian.PutUint32(stored[rows*inner+4*row:], math.Float32bits(scale))
			}
			decoded := testcheck.Must(decode(stored, shape.rows, shape.inner, uint32(int8ConvRotGroup)))(t)
			host := make([]float32, inner)
			parted := 0
			for row := range rows {
				scale := math.Float32frombits(binary.LittleEndian.Uint32(stored[rows*inner+4*row:]))
				decodeRowF32(host, stored[row*inner:(row+1)*inner], scale, int8ConvRotGroup)
				for column, want := range host {
					got := math.Float32frombits(binary.LittleEndian.Uint32(decoded[4*(row*inner+uint64(column)):]))
					if math.Float32bits(got) != math.Float32bits(want) && !(got != got && want != want) {
						if parted++; parted <= 4 {
							t.Errorf("row %d (scale %g), column %d decodes to %g (%#08x) on the card, %g (%#08x) on the host", row, scale, column, got, math.Float32bits(got), want, math.Float32bits(want))
						}
					}
				}
			}
			if parted > 0 {
				t.Errorf("%d of %d values part from the host's decode; want none", parted, rows*inner)
			}
		})
	}
	for _, group := range []uint32{64, 1024} {
		if _, err := decode(make([]byte, 4096+4), 1, 4096, group); err == nil {
			t.Errorf("a decode in groups of %d was taken; want it refused", group)
		}
	}
}

// decodeRowF32 decodes one row of an int8 ConvRot weight into float32 as comfy_kitchen's CUDA dequantize decodes it into
// a float32 input's dtype, which ComfyUI's text encoders run in: each value its int8 times scale, the group of size
// columns turned by radix-4 stages over strides 1, 4, 16 and on, each output half the left-to-right sum of its four
// signed inputs, every operation rounded to float32, and nothing rounded further. It was ComfyUI's bit for bit on the
// MOSS Ultra encoder's weights, 3.9 M elements of six (moss-encoder-own).
func decodeRowF32(out []float32, quantized []byte, scale float32, size uint64) {
	for start := uint64(0); start < uint64(len(quantized)); start += size {
		values := out[start : start+size]
		for index := range values {
			values[index] = float32(int8(quantized[start+uint64(index)])) * scale
		}
		for stride := uint64(1); stride < size; stride *= 4 {
			for block := uint64(0); block < size; block += 4 * stride {
				for offset := block; offset < block+stride; offset++ {
					x0, x1, x2, x3 := values[offset], values[offset+stride], values[offset+2*stride], values[offset+3*stride]
					values[offset] = 0.5 * (x0 + x1 + x2 - x3)
					values[offset+stride] = 0.5 * (x0 + x1 - x2 + x3)
					values[offset+2*stride] = 0.5 * (x0 - x1 + x2 + x3)
					values[offset+3*stride] = 0.5 * (x1 - x0 + x2 + x3)
				}
			}
		}
	}
}
